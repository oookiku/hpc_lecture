#include <cassert>
#include <cstdio>
#include <chrono>
#include <vector>
#include "hdf5.h"
using namespace std;

void outputBlockPattern(int *data, int NlocalX, int NlocalY, int rank);

int main (int argc, char** argv) {
  hsize_t dim[2] = {2, 2};
  int mpisize, mpirank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  assert(mpisize == dim[0]*dim[1]);
  hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file = H5Fopen("data.h5", H5F_ACC_RDONLY, plist);
  hid_t dataset = H5Dopen(file, "dataset", H5P_DEFAULT);
  hid_t globalspace = H5Dget_space(dataset);
  int ndim = H5Sget_simple_extent_ndims(globalspace);
  hsize_t N[ndim];
  H5Sget_simple_extent_dims(globalspace, N, NULL);
  hsize_t NX = N[0], NY = N[1];
  hsize_t Nlocal[2] = {NX/dim[0], NY/dim[1]};
  hsize_t offset[2] = {mpirank / dim[0], mpirank % dim[0]};
  for(int i=0; i<2; i++) offset[i] *= Nlocal[i];
  hsize_t count[2] = {1,1};
  hsize_t stride[2] = {1,1};
  hid_t localspace = H5Screate_simple(2, Nlocal, NULL);
  H5Sselect_hyperslab(globalspace, H5S_SELECT_SET, offset, stride, count, Nlocal);
  H5Pclose(plist);
  vector<int> buffer(Nlocal[0]*Nlocal[1]);
  plist = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
  H5Dread(dataset, H5T_NATIVE_INT, localspace, globalspace, plist, &buffer[0]);
  H5Dclose(dataset);
  H5Sclose(localspace);
  H5Sclose(globalspace);
  H5Fclose(file);
  H5Pclose(plist);
  outputBlockPattern(&buffer[0], Nlocal[1], Nlocal[0], mpirank);
  MPI_Finalize();
}

void outputBlockPattern(
  int *data,
  int NlocalX,
  int NlocalY, 
  int rank
)
{
  std::string scalar = "";
  for (int j = NlocalY-1; j >= 0; j--) {
    for (int i = 0; i < NlocalX; i++) {
      scalar += std::to_string(data[i+NlocalX*j]);
      scalar += " ";
    }
  }
  std::string fname = "result" + std::to_string(rank) + ".vti";
  FILE *fp = fopen(fname.c_str(), "w");
  int offset[2] = {1-rank/2, rank%2};
  // header 
  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"ImageData\" version=\"0.1\">\n");
  fprintf(fp, "\t<ImageData WholeExtent=\"%d %d %d %d 0 0\" Origin=\"0 0 0\" Space=\"%f %f %f\">\n",
              NlocalX*offset[1], NlocalX*(offset[1]+1),
              NlocalY*offset[0], NlocalY*(offset[0]+1),
              1.0, 1.0, 0.0);
  fprintf(fp, "\t\t<Piece Extent=\"%d %d %d %d 0 0\">\n",
              NlocalX*offset[1], NlocalX*(offset[1]+1),
              NlocalY*offset[0], NlocalY*(offset[0]+1));
  // cell data
  fprintf(fp, "\t\t\t<CellData>\n");
  fprintf(fp, "\t\t\t\t<DataArray Name = \"rank\" type=\"Int32\" format=\"ascii\">\n");
  fprintf(fp, scalar.c_str());
  fprintf(fp, "\n\t\t\t\t</DataArray>\n");
  fprintf(fp, "\t\t\t</CellData>\n");
  // last elements
  fprintf(fp, "\t\t</Piece>\n");
  fprintf(fp, "\t</ImageData>\n");
  fprintf(fp, "</VTKFile>\n");
  fclose(fp);
}
