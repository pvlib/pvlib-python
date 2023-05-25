"""
Get .PAN or .OND file data into a nested dictionary.
"""

import io

def num_type(value):
    # Determine if a value is float, int or leave as string

    if '.' in value:
        try:  # Detect float
            value_out = float(value)
            return value_out
        
        except ValueError:  # Otherwise leave as string
            value_out = value
            return value_out
    
    else:

        try:  # Detect int
            value_out = int(value)
            return value_out
        
        except ValueError:  # Otherwise leave as string
            value_out = value
            return value_out
    

def element_type(element):
    # Determine if an element is a list then pass to num_type()

    if ',' in element:  # Detect a list
        values = element.split(',')
        element_out = []
        for val in values:  # Determine datatype of each value
            element_out.append(num_type(val))
        
        return element_out

    else:
        return num_type(element)


def parse_panond(fbuf):
    """
    Parse a .pan or .ond text file into a nested dictionary.

    Parameters
    ----------
    fbuf : File-like object
        Buffer of a .pan or .ond file
    
    Returns
    -------
    comp : Nested Dictionary
        Contents of the .pan or .ond file following the indentation of the file. 
        The value of datatypes are assumed during reading. The value units are
        the default used by PVsyst.

    Raises
    ------

    Notes
    -----

    See Also
    --------

    References
    ----------
    """
    comp = {}  # Component
    dict_levels = [comp]
    
    fbuf.seek(0)
    lines = fbuf.getvalue().splitlines()
    for i in range(0, len(lines) - 1):  # Reading blank lines. Stopping one short to avoid index error. Last line never contains important data.
        if lines[i] == '':  # Skipping blank lines
              continue

        indent_lvl_1 = (len(lines[i]) - len(lines[i].lstrip(' '))) // 2 
        indent_lvl_2 = (len(lines[i + 1]) - len(lines[i + 1].lstrip(' '))) // 2 
        line_data = lines[i].split('=') 
        key = line_data[0].strip()
        value = element_type(line_data[1].strip()) if len(line_data) > 1 else None
        if indent_lvl_2 > indent_lvl_1:  # add a level to the dict. The key here will be ignored. Not vital to file function.
            current_level = dict_levels[indent_lvl_1]
            new_level = {}
            current_level[key] = new_level
            dict_levels = dict_levels[: indent_lvl_1 + 1] + [new_level]
            current_level = dict_levels[indent_lvl_1 + 1]
            current_level[key] = value

        elif indent_lvl_2 <= indent_lvl_1:  # add key/value to dict
            current_level = dict_levels[indent_lvl_1]
            current_level[key] = value

    return comp


def read_panond(file):
    """
    Retrieve Module or Inverter data from a .pan or .ond text file, respectively.

    Parameters
    ----------
    file : string or path object
        Name or path of a .pan/.ond file
    
    Returns
    -------
    content : Nested Dictionary
        Contents of the .pan or .ond file following the indentation of the file. 
        The value of datatypes are assumed during reading. The value units are
        the default used by PVsyst.

    Raises
    ------

    Notes
    -----

    See Also
    --------

    References
    ----------
    """

    with open(file, "r", encoding='utf-8-sig') as file:
        f_content = file.read()
        fbuf = io.StringIO(f_content)
    
    content = parse_panond(fbuf)

    return content
