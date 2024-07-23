#ifndef HDF_WRAPPER_INCLUDED
#define HDF_WRAPPER_INCLUDED

#include "hdf5.h"
#include "hdf5_hl.h"
// #include "H5Cpp.h"
#include <iostream>
#include <mpi.h>

#ifdef HBT_REAL8
#define H5T_HBTReal H5T_NATIVE_DOUBLE
#else
#define H5T_HBTReal H5T_NATIVE_FLOAT
#endif
#ifdef HBT_INT8
#define H5T_HBTInt H5T_NATIVE_LONG
#else
#define H5T_HBTInt H5T_NATIVE_INT
#endif

extern void writeHDFmatrix(hid_t file, const void *buf, const char *name, hsize_t ndim, const hsize_t *dims,
                           hid_t dtype, hid_t dtype_file);

inline int GetDatasetDims(hid_t dset, hsize_t dims[])
{
  hid_t dspace = H5Dget_space(dset);
  int ndim = H5Sget_simple_extent_dims(dspace, dims, NULL);
  H5Sclose(dspace);
  return ndim;
}

inline herr_t ReclaimVlenData(hid_t dset, hid_t dtype, void *buf)
{
  herr_t status;
  hid_t dspace = H5Dget_space(dset);
  status = H5Dvlen_reclaim(dtype, dspace, H5P_DEFAULT, buf);
  status = H5Sclose(dspace);
  return status;
}

inline herr_t ReadDataset(hid_t file, const char *name, hid_t dtype, void *buf)
/* read named dataset from file into buf.
 * dtype specifies the datatype of buf; it does not need to be the same as the storage type in file*/
{
  herr_t status;
  hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
  status = H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
  if (status < 0)
  {
    const int bufsize = 1024;
    char grpname[bufsize], filename[bufsize];
    H5Iget_name(file, grpname, bufsize);
    H5Fget_name(file, filename, bufsize);

    throw std::runtime_error("#### ERROR READING DATASET " + std::string(grpname) + "/" + std::string(name) + " from " + std::string(filename) + ", error number " + std::to_string(status));
  }
  H5Dclose(dset);
  return status;
}

inline herr_t ReadPartialDataset(hid_t file, const char *name, hid_t dtype, void *buf, hsize_t offset, hsize_t count)
/* read named dataset from file into buf.
 * dtype specifies the datatype of buf; it does not need to be the same as the storage type in file
 * offset and count specify the range of elements in the first dimension to read */
{
  herr_t status;
  hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);

  /* Get dataspace in the file */
  hid_t file_space_id = H5Dget_space(dset);

  /* Get size of dataset in the file */
  const int max_dims = 32;
  hsize_t dims[max_dims];
  int rank = H5Sget_simple_extent_ndims(file_space_id);
  H5Sget_simple_extent_dims(file_space_id, dims, NULL);

  /* Create memory dataspace same size as in the file, except that first dimension has count elements */
  dims[0] = count;
  hsize_t mem_space_id = H5Screate_simple(rank, dims, NULL);

  /* Select elements in the file to read */
  hsize_t start_arr[max_dims];
  hsize_t count_arr[max_dims];
  start_arr[0] = offset;
  count_arr[0] = count;
  for (int i = 1; i < rank; i += 1)
  {
    start_arr[i] = 0;
    count_arr[i] = dims[i];
  }
  H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET, start_arr, NULL, count_arr, NULL);

  /* Read the data */
  status = H5Dread(dset, dtype, mem_space_id, file_space_id, H5P_DEFAULT, buf);
  if (status < 0)
  {
    const int bufsize = 1024;
    char grpname[bufsize], filename[bufsize];
    H5Iget_name(file, grpname, bufsize);
    H5Fget_name(file, filename, bufsize);

    throw std::runtime_error("#### ERROR READING DATASET " + std::string(grpname) + "/" + std::string(name) + " from " + std::string(filename) + ", error number " + std::to_string(status));
  }
  H5Dclose(dset);
  H5Sclose(mem_space_id);
  H5Sclose(file_space_id);

  return status;
}

inline herr_t ReadAttribute(hid_t loc_id, const char *obj_name, const char *attr_name, hid_t dtype, void *buf)
/* read named attribute of object into buf. if loc_id fully specifies the object, obj_name="."
 * dtype specifies the datatype of buf; it does not need to be the same as the storage type in file*/
{
  herr_t status;
  hid_t attr = H5Aopen_by_name(loc_id, obj_name, attr_name, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Aread(attr, dtype, buf);
  status = H5Aclose(attr);

  if (status < 0)
  {
    const int bufsize = 1024;
    char grpname[bufsize], filename[bufsize];
    H5Fget_name(loc_id , filename, bufsize);

    throw std::runtime_error("#### ERROR READING ATTRIBUTE " + std::string(obj_name) + "/" + std::string(attr_name) + " from " + std::string(filename) + ", error number " + std::to_string(status));
  }

  return status;
}

/* As above, but for string attributes */
inline herr_t ReadAttribute(hid_t loc_id, const char *obj_name, const char *attr_name, std::string &buf)
{
  herr_t status;

  // Open the attribute and determine length of the string
  hid_t attr = H5Aopen_by_name(loc_id, obj_name, attr_name, H5P_DEFAULT, H5P_DEFAULT);
  hid_t dtype = H5Aget_type(attr);

  // Create memory buffer to read into and type to describe it
  const size_t maxlen = 1024;
  char readbuf[maxlen];
  hid_t mem_type = H5Tcreate(H5T_STRING, maxlen);
  H5Tset_strpad(mem_type, H5T_STR_NULLTERM);

  // Read the attribute
  status = H5Aread(attr, mem_type, readbuf);

  // Assign value to the output string
  buf = readbuf;

  H5Tclose(dtype);
  H5Tclose(mem_type);
  H5Aclose(attr);

  /* An error has occured. Terminate the program. */
  if(status < 0)
  {
    const int bufsize = 1024;
    char grpname[bufsize], filename[bufsize];
    H5Fget_name(loc_id , filename, bufsize);

    throw std::runtime_error("#### ERROR READING ATTRIBUTE " + std::string(obj_name) + "/" + std::string(attr_name) + " from " + std::string(filename) + ", error number " + std::to_string(status));
  }

  return status;
}

inline void writeHDFmatrix(hid_t file, const void *buf, const char *name, hsize_t ndim, const hsize_t *dims,
                           hid_t dtype)
{
  writeHDFmatrix(file, buf, name, ndim, dims, dtype, dtype);
}

void writeStringAttribute(hid_t handle, const char *buf, const char *attr_name);

#endif
