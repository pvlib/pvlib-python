"""
Older versions of PAN files created by PVsyst use a Borland Pascal Real48
format.

This is based on:
    https://github.com/CanadianSolar/CASSYS/blob/
    b5487bb4e9e77174c805d64e3c960c46d357b7e2/CASSYS%20Interface/
    DatabaseImportModule.vba#L4
"""

import struct

# --- Constants ---
SEMICOLON_MARKER = 0x3B
DOT_MARKER = 0x09
DOUBLE_DOT_MARKER = 0x0A
FORWARD_SLASH_MARKER = 0x2F
CR_MARKER = 0x0D  # Carriage Return
VERTICAL_BAR_MARKER = 0xA6


# --- Supporting Functions ---
def _read48_to_float(real48):
    """
    Convert a 6-byte Delphi Real48 encoded value to a standard Python float.

    Parameters
    ----------
    real48 : bytes
        6-byte Delphi Real48 encoded value

    Returns
    -------
    value : float
        Converted float value

    Notes
    -----
    The format consists of:
    - 1 byte: Exponent (offset by 129)
    - 5 bytes: Mantissa, with the last bit of the 5th byte as the sign bit.
    """
    if not real48 or len(real48) != 6 or real48[0] == 0:
        return 0.0

    # The exponent is the first byte, with an offset of 129
    exponent = float(real48[0] - 129)

    mantissa = 0.0

    # Process the first 4 bytes of the mantissa
    # The division by 256 (or multiplication by 0.00390625) shifts the bytes
    for i in range(4, 0, -1):
        mantissa += real48[i]
        mantissa /= 256.0

    # Process the 5th byte of the mantissa
    mantissa += real48[5] & 0x7F  # Use only the first 7 bits
    mantissa /= 128.0  # equivalent to * 0.0078125
    mantissa += 1.0

    # Check the sign bit (the last bit of the 6th byte)
    if (real48[5] & 0x80) == 0x80:
        mantissa = -mantissa

    # Final calculation using the exponent
    return mantissa * (2.0**exponent)


def _find_marker_index(marker, start_index, byte_array):
    """
    Find the index of the first occurrence of a hex marker after a start index.

    Parameters
    ----------
    marker : int
        Hex marker to search for
    start_index : int
        Starting index for the search
    byte_array : bytes
        Byte array to search in

    Returns
    -------
    index : int
        Index right after the marker, or raises ValueError if not found
    """
    # bytearray.find is more efficient than a manual loop
    found_index = byte_array.find(bytes([marker]), start_index)
    if found_index != -1:
        return found_index + 1
    if found_index is None:
        raise ValueError(f"Marker {marker} is not in byte array")
    return found_index


def _get_param_index(start_index, offset_num):
    """
    Calculate the start index of a Real48 parameter.

    Parameters
    ----------
    start_index : int
        Starting index of the Real48 data block
    offset_num : int
        Offset number of the parameter

    Returns
    -------
    index : int
        Start index of the Real48 parameter
    """
    return start_index + 6 * offset_num


def _extract_byte_parameters(byte_array, start_index, num_bytes):
    """
    Extract bytes that form a single parameter from the original byte array.

    Parameters
    ----------
    byte_array : bytes
        Original byte array containing the whole file
    start_index : int
        Starting index for extraction
    num_bytes : int
        Number of bytes to extract

    Returns
    -------
    param_bytes : bytes
        Extracted byte sequence forming a single parameter
    """
    # Check bounds to avoid index errors
    if start_index + num_bytes > len(byte_array):
        raise IndexError(
            f"Not enough bytes: need {num_bytes} bytes starting at "
            f"{start_index}"
        )

    # Extract the specified number of bytes starting at start_index
    param_byte_sequence = byte_array[start_index:start_index + num_bytes]

    return param_byte_sequence


# This format might be specific to how PAN files format their floats
value_format = "{:.2f}"


def _extract_iam_profile(start_index, byte_array):
    """
    Extract the IAM (Incidence Angle Modifier) profile.

    Parameters
    ----------
    start_index : int
        Starting index of the IAM data in the byte array
    byte_array : bytes
        Byte array containing the file data

    Returns
    -------
    iam_profile : list of dict
        List of dictionaries containing 'aoi' and 'modifier' values
    """
    iam_profile = []

    for i in range(0, 45, 5):  # 0 to 44 step 5 (matches VB.NET loop)
        # Extract AOI value
        aoi_index = _get_param_index(start_index=start_index, offset_num=i)
        aoi_bytes = _extract_byte_parameters(
            byte_array=byte_array, start_index=aoi_index, num_bytes=6
        )
        aoi_raw = _read48_to_float(real48=aoi_bytes)
        aoi_formatted = value_format.format(aoi_raw)  # Keep for the check

        # Check if AOI is not null/empty (like VB.NET vbNullString check)
        if aoi_formatted != "":
            # Extract modifier value
            modifier_index = _get_param_index(
                start_index=start_index, offset_num=i + 1
            )
            modifier_bytes = _extract_byte_parameters(
                byte_array=byte_array, start_index=modifier_index, num_bytes=6
            )
            modifier_raw = _read48_to_float(real48=modifier_bytes)

            # Add to profile (only if AOI is not empty)
            iam_profile.append({"aoi": aoi_raw, "modifier": modifier_raw})
        # If AOI is empty, we skip this entry entirely (don't add to list)
    return iam_profile


def read_pan_binary(filename):
    """
    Retrieve Module data from a .pan binary file.

    Parameters
    ----------
    filename : str or path object
        Name or path of a .pan binary file

    Returns
    -------
    content : dict
        Contents of the .pan file.

    Notes
    -----
    The parser is intended for use with binary .pan files that were created for
    PVsyst version 6.39 or earlier. At time of publication, no documentation
    for these files was available. So, this parser is based on inferred logic,
    rather than anything specified by PVsyst.

    The parser can only be used on binary .pan files.
    For files that use the newer text format
    please refer to `pvlib.iotools.panond.read_panond`.

    """
    data = {}

    # Read the file and convert to byte array
    with open(filename, "rb") as file:
        byte_array = file.read()

    if not byte_array:
        raise ValueError("File is empty")

    # --- Find start indices for string parameters ---
    try:
        manu_start_index = _find_marker_index(
            marker=SEMICOLON_MARKER, start_index=0, byte_array=byte_array
        )
        panel_start_index = _find_marker_index(
            marker=DOT_MARKER, start_index=0, byte_array=byte_array
        )
        source_start_index = _find_marker_index(
            marker=DOT_MARKER,
            start_index=panel_start_index,
            byte_array=byte_array
        )
        version_start_index = _find_marker_index(
            marker=DOUBLE_DOT_MARKER,
            start_index=source_start_index,
            byte_array=byte_array,
        )
        version_end_index = _find_marker_index(
            marker=SEMICOLON_MARKER,
            start_index=version_start_index,
            byte_array=byte_array,
        )
        year_start_index = _find_marker_index(
            marker=SEMICOLON_MARKER,
            start_index=version_end_index,
            byte_array=byte_array,
        )
        technology_start_index = _find_marker_index(
            marker=DOUBLE_DOT_MARKER,
            start_index=year_start_index,
            byte_array=byte_array,
        )
        cells_in_series_start_index = _find_marker_index(
            marker=SEMICOLON_MARKER,
            start_index=technology_start_index,
            byte_array=byte_array,
        )
        cells_in_parallel_start_index = _find_marker_index(
            marker=SEMICOLON_MARKER,
            start_index=cells_in_series_start_index,
            byte_array=byte_array,
        )
        bypass_diodes_start_index = _find_marker_index(
            marker=SEMICOLON_MARKER,
            start_index=cells_in_parallel_start_index,
            byte_array=byte_array,
        )

        # --- Find start of Real48 encoded data ---
        cr_counter = 0
        real48_start_index = 0
        for i, byte in enumerate(byte_array):
            if byte == CR_MARKER:
                cr_counter += 1
            if cr_counter == 3:
                real48_start_index = i + 2  # Skip <CR><LF>
                break

        if real48_start_index == 0:
            return {"error": "Could not find start of Real48 data block."}

        # --- Extract string parameters ---
        # Note: latin-1 is used as it can decode any byte value without error
        data["Manufacturer"] = (
            byte_array[manu_start_index: panel_start_index - 1]
            .decode("latin-1")
            .strip()
        )
        data["Model"] = (
            byte_array[panel_start_index: source_start_index - 1]
            .decode("latin-1")
            .strip()
        )
        data["Source"] = (
            byte_array[source_start_index: version_start_index - 4]
            .decode("latin-1")
            .strip()
        )
        data["Version"] = (
            byte_array[version_start_index: version_end_index - 2]
            .decode("latin-1")
            .replace("Version", "PVsyst")
            .strip()
        )
        data["Year"] = (
            byte_array[year_start_index: year_start_index + 4]
            .decode("latin-1")
            .strip()
        )
        data["Technology"] = (
            byte_array[
                technology_start_index: cells_in_series_start_index - 1
            ]
            .decode("latin-1")
            .strip()
        )
        data["Cells_In_Series"] = (
            byte_array[
                cells_in_series_start_index: cells_in_parallel_start_index - 1
            ]
            .decode("latin-1")
            .strip()
        )
        data["Cells_In_Parallel"] = (
            byte_array[
                cells_in_parallel_start_index: bypass_diodes_start_index - 1
            ]
            .decode("latin-1")
            .strip()
        )

        # --- Parse Real48 encoded parameters ---
        param_map = {
            "PNom": 0,
            "VMax": 1,
            "Tolerance": 2,
            "AreaM": 3,
            "CellArea": 4,
            "GRef": 5,
            "TRef": 6,
            "Isc": 8,
            "muISC": 9,
            "Voc": 10,
            "muVocSpec": 11,
            "Imp": 12,
            "Vmp": 13,
            "BypassDiodeVoltage": 14,
            "RShunt": 17,
            "RSerie": 18,
            "RShunt_0": 23,
            "RShunt_exp": 24,
            "muPmp": 25,
        }

        for name, offset in param_map.items():
            start = _get_param_index(
                start_index=real48_start_index, offset_num=offset
            )
            end = start + 6
            param_bytes = byte_array[start:end]
            value = _read48_to_float(real48=param_bytes)
            if name == "Tolerance":
                value *= 100  # Convert to percentage
                if value > 100:
                    value = 0.0
            data[name] = value

        # --- Check for and Parse IAM Profile ---
        dot_counter = 0
        iam_start_index = 0
        dot_position = data["Version"].find(".")
        major_version = int(data["Version"][dot_position - 1: dot_position])
        if major_version < 6:
            for i in range(real48_start_index + 170, len(byte_array)):
                if byte_array[i] == DOT_MARKER:
                    dot_counter += 1
                if dot_counter == 2:
                    iam_start_index = i + 4
                    break

        if iam_start_index > 0:
            data["IAMProfile"] = _extract_iam_profile(
                start_index=iam_start_index, byte_array=byte_array
            )

    except (IndexError, TypeError, struct.error) as e:
        return {"error": f"Failed to parse binary PAN file: {e}"}

    return data
