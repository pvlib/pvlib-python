#Created with SAM version 2018.11.11
# Modified by C. W. Hansen for PEP8 compliance
import sys
import os
import ctypes
ctypes.c_number = ctypes.c_float # must be c_double or c_float depending on how defined in sscapi.h
class PySSC():
    def __init__(self, sam_dir):
        if sys.platform in ['win32', 'cygwin']:
            self.pdll = ctypes.CDLL(os.path.join(sam_dir, "ssc.dll"))
        elif sys.platform == 'darwin':
            self.pdll = ctypes.CDLL(os.path.join(sam_dir, "ssc.dylib"))
        elif sys.platform == 'linux2':
            self.pdll = ctypes.CDLL(os.path.join(sam_dir, "ssc.so"))   # instead of relative path, require user to have on LD_LIBRARY_PATH
        else:
            print ('Platform not supported ', sys.platform)
    INVALID=0
    STRING=1
    NUMBER=2
    ARRAY=3
    MATRIX=4
    INPUT=1
    OUTPUT=2
    INOUT=3

    def version(self):
        self.pdll.ssc_version.restype = ctypes.c_int
        return self.pdll.ssc_version()

    def build_info(self):
        self.pdll.ssc_build_info.restype = ctypes.c_char_p
        return self.pdll.ssc_build_info()

    def data_create(self):
        self.pdll.ssc_data_create.restype = ctypes.c_void_p
        return self.pdll.ssc_data_create()

    def data_free(self, p_data):
        self.pdll.ssc_data_free(ctypes.c_void_p(p_data))

    def data_clear(self, p_data):
        self.pdll.ssc_data_clear(ctypes.c_void_p(p_data))

    def data_unassign(self, p_data, name):
        self.pdll.ssc_data_unassign(ctypes.c_void_p(p_data),
                                    ctypes.c_char_p(name))

    def data_query(self, p_data, name):
        self.pdll.ssc_data_query.restype = ctypes.c_int
        return self.pdll.ssc_data_query(ctypes.c_void_p(p_data),
                                        ctypes.c_char_p(name))

    def data_first(self, p_data):
        self.pdll.ssc_data_first.restype = ctypes.c_char_p
        return self.pdll.ssc_data_first(ctypes.c_void_p(p_data))

    def data_next(self, p_data):
        self.pdll.ssc_data_next.restype = ctypes.c_char_p
        return self.pdll.ssc_data_next(ctypes.c_void_p(p_data))

    def data_set_string(self, p_data, name, value):
        self.pdll.ssc_data_set_string(ctypes.c_void_p(p_data),
                                      ctypes.c_char_p(name),
                                      ctypes.c_char_p(value))

    def data_set_number(self, p_data, name, value):
        self.pdll.ssc_data_set_number(ctypes.c_void_p(p_data),
                                      ctypes.c_char_p(name),
                                      ctypes.c_number(value))

    def data_set_array(self,p_data,name,parr):
        count = len(parr)
        arr = (ctypes.c_number * count)()
        arr[:] = parr # set all at once instead of looping
        return self.pdll.ssc_data_set_array(ctypes.c_void_p(p_data),
                                            ctypes.c_char_p(name),
                                            ctypes.pointer(arr),
                                            ctypes.c_int(count))

    def data_set_array_from_csv(self, p_data, name, fn):
        f = open(fn, 'rb');
        data = [];
        for line in f :
            data.extend([n for n in map(float, line.split(b','))])
        f.close()
        return self.data_set_array(p_data, name, data)

    def data_set_matrix(self,p_data,name,mat):
        nrows = len(mat)
        ncols = len(mat[0])
        size = nrows * ncols
        arr = (ctypes.c_number * size)()
        idx=0
        for r in range(nrows):
            for c in range(ncols):
                arr[idx] = ctypes.c_number(mat[r][c])
                idx += 1
        return self.pdll.ssc_data_set_matrix(ctypes.c_void_p(p_data),
                                             ctypes.c_char_p(name),
                                             ctypes.pointer(arr),
                                             ctypes.c_int(nrows),
                                             ctypes.c_int(ncols))

    def data_set_matrix_from_csv(self, p_data, name, fn):
        f = open(fn, 'rb');
        data = [];
        for line in f :
            lst = ([n for n in map(float, line.split(b','))])
            data.append(lst);
        f.close();
        return self.data_set_matrix(p_data, name, data);

    def data_set_table(self,p_data,name,tab):
        return self.pdll.ssc_data_set_table(ctypes.c_void_p(p_data),
                                            ctypes.c_char_p(name),
                                            ctypes.c_void_p(tab));

    def data_get_string(self, p_data, name):
        self.pdll.ssc_data_get_string.restype = ctypes.c_char_p
        return self.pdll.ssc_data_get_string(ctypes.c_void_p(p_data),
                                             ctypes.c_char_p(name))

    def data_get_number(self, p_data, name):
        val = ctypes.c_number(0)
        self.pdll.ssc_data_get_number(ctypes.c_void_p(p_data),
                                      ctypes.c_char_p(name),
                                      ctypes.byref(val))
        return val.value

    def data_get_array(self,p_data,name):
        count = ctypes.c_int()
        self.pdll.ssc_data_get_array.restype = ctypes.POINTER(ctypes.c_number)
        parr = self.pdll.ssc_data_get_array(ctypes.c_void_p(p_data),
                                            ctypes.c_char_p(name),
                                            ctypes.byref(count))
        arr = parr[0:count.value] # extract all at once
        return arr

    def data_get_matrix(self,p_data,name):
        nrows = ctypes.c_int()
        ncols = ctypes.c_int()
        self.pdll.ssc_data_get_matrix.restype = ctypes.POINTER(ctypes.c_number)
        parr = self.pdll.ssc_data_get_matrix(ctypes.c_void_p(p_data),
                                             ctypes.c_char_p(name),
                                             ctypes.byref(nrows),
                                             ctypes.byref(ncols))
        idx = 0
        mat = []
        for r in range(nrows.value):
            row = []
            for c in range(ncols.value):
                row.append( float(parr[idx]) )
                idx = idx + 1
            mat.append(row)
        return mat
    # don't call data_free() on the result, it's an internal
    # pointer inside SSC

    def data_get_table(self,p_data,name):
        return self.pdll.ssc_data_get_table(ctypes.c_void_p(p_data), name);

    def module_entry(self,index):
        self.pdll.ssc_module_entry.restype = ctypes.c_void_p
        return self.pdll.ssc_module_entry(ctypes.c_int(index))

    def entry_name(self,p_entry):
        self.pdll.ssc_entry_name.restype = ctypes.c_char_p
        return self.pdll.ssc_entry_name(ctypes.c_void_p(p_entry))

    def entry_description(self,p_entry):
        self.pdll.ssc_entry_description.restype = ctypes.c_char_p
        return self.pdll.ssc_entry_description(ctypes.c_void_p(p_entry))

    def entry_version(self,p_entry):
        self.pdll.ssc_entry_version.restype = ctypes.c_int
        return self.pdll.ssc_entry_version(ctypes.c_void_p(p_entry))

    def module_create(self,name):
        self.pdll.ssc_module_create.restype = ctypes.c_void_p
        return self.pdll.ssc_module_create(ctypes.c_char_p(name))

    def module_free(self,p_mod):
        self.pdll.ssc_module_free(ctypes.c_void_p(p_mod))

    def module_var_info(self,p_mod,index):
        self.pdll.ssc_module_var_info.restype = ctypes.c_void_p
        return self.pdll.ssc_module_var_info(ctypes.c_void_p(p_mod),
                                             ctypes.c_int(index))

    def info_var_type(self, p_inf):
        return self.pdll.ssc_info_var_type(ctypes.c_void_p(p_inf))

    def info_data_type(self, p_inf):
        return self.pdll.ssc_info_data_type(ctypes.c_void_p(p_inf))

    def info_name(self, p_inf):
        self.pdll.ssc_info_name.restype = ctypes.c_char_p
        return self.pdll.ssc_info_name(ctypes.c_void_p(p_inf))

    def info_label(self, p_inf):
        self.pdll.ssc_info_label.restype = ctypes.c_char_p
        return self.pdll.ssc_info_label(ctypes.c_void_p(p_inf))

    def info_units(self, p_inf):
        self.pdll.ssc_info_units.restype = ctypes.c_char_p
        return self.pdll.ssc_info_units(ctypes.c_void_p(p_inf))

    def info_meta(self, p_inf):
        self.pdll.ssc_info_meta.restype = ctypes.c_char_p
        return self.pdll.ssc_info_meta(ctypes.c_void_p(p_inf))

    def info_group(self, p_inf):
        self.pdll.ssc_info_group.restype = ctypes.c_char_p
        return self.pdll.ssc_info_group(ctypes.c_void_p(p_inf))

    def info_uihint(self, p_inf):
        self.pdll.ssc_info_uihint.restype = ctypes.c_char_p
        return self.pdll.ssc_info_uihint(ctypes.c_void_p(p_inf))

    def info_required(self, p_inf):
        self.pdll.ssc_info_required.restype = ctypes.c_char_p
        return self.pdll.ssc_info_required(ctypes.c_void_p(p_inf))

    def info_constraints(self, p_inf):
        self.pdll.ssc_info_constraints.restype = ctypes.c_char_p
        return self.pdll.ssc_info_constraints(ctypes.c_void_p(p_inf))

    def module_exec(self, p_mod, p_data):
        self.pdll.ssc_module_exec.restype = ctypes.c_int
        return self.pdll.ssc_module_exec(ctypes.c_void_p(p_mod),
                                         ctypes.c_void_p(p_data))

    def module_exec_simple_no_thread(self, modname, data):
        self.pdll.ssc_module_exec_simple_nothread.restype = ctypes.c_char_p;
        return self.pdll.ssc_module_exec_simple_nothread(
            ctypes.c_char_p(modname), ctypes.c_void_p(data));

    def module_log(self, p_mod, index):
        log_type = ctypes.c_int()
        time = ctypes.c_float()
        self.pdll.ssc_module_log.restype = ctypes.c_char_p
        return self.pdll.ssc_module_log(ctypes.c_void_p(p_mod),
                                        ctypes.c_int(index),
                                        ctypes.byref(log_type),
                                        ctypes.byref(time))

    def module_exec_set_print(self, prn):
        return self.pdll.ssc_module_exec_set_print(ctypes.c_int(prn));
