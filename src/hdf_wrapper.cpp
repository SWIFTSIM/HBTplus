#include "hdf_wrapper.h"

void writeHDFmatrix(hid_t file, const void *buf, const char *name, hsize_t ndim, const hsize_t *dims, hid_t dtype,
                    hid_t dtype_file)
{
  hid_t dataspace = H5Screate_simple(ndim, dims, NULL);
  hid_t dataset = H5Dcreate2(file, name, dtype_file, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (!(NULL == buf || 0 == dims[0]))
  {
    herr_t status = H5Dwrite(dataset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    if (status < 0)
    {
      const int bufsize = 1024;
      char grpname[bufsize], filename[bufsize];
      H5Iget_name(file, grpname, bufsize);
      H5Fget_name(file, filename, bufsize);
      std::cerr << "####ERROR WRITING " << grpname << "/" << name << " into " << filename << ", error number " << status
                << std::endl
                << std::flush;
    }
  }
  H5Dclose(dataset);
  H5Sclose(dataspace);
}

/* Writes a string attribute in the specified file handle (e.g group, dataset, etc)*/
void writeStringAttribute(hid_t handle, const char *buf, const char *attr_name)
{
  if (H5Aexists(handle, attr_name))
    H5Adelete(handle, attr_name);

  hid_t atype = H5Tcopy(H5T_C_S1);
  herr_t status = H5Tset_size(atype, strlen(buf));

  hid_t hdf5_dataspace = H5Screate(H5S_SCALAR);
  hid_t hdf5_attribute = H5Acreate(handle, attr_name, atype, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);

  status = H5Awrite(hdf5_attribute, atype, buf);

  status = H5Aclose(hdf5_attribute);
  status = H5Sclose(hdf5_dataspace);
}