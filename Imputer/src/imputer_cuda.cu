#include "imputer.h"
using namespace torch::indexing;
using namespace std;
__global__ void path_counter(int turn, torch::PackedTensorAccessor32<double,3> grid, int max_z, int max_y, int max_x){
    int x_idx = blockDim.x*blockIdx.x + threadIdx.x;
    int y_idx = blockDim.y*blockIdx.y + threadIdx.y;
    int z_idx = turn - x_idx - y_idx;

    if(x_idx>=max_x || y_idx>=max_y || z_idx>=max_z || z_idx<0) return;
    
    int sum = 0;
    if (x_idx>0){
        sum += x_idx*grid[z_idx][y_idx][x_idx-1];
    }
    if (y_idx>0){
        sum += y_idx*grid[z_idx][y_idx-1][x_idx];
    }
    if (z_idx>0){
        sum += z_idx*grid[z_idx-1][y_idx][x_idx];
    }

    grid[z_idx][y_idx][x_idx]*=(sum/turn);



};

torch::Tensor process_octant(torch::Tensor cuboid){
    torch::PackedTensorAccessor32<double,3> grid = cuboid.packed_accessor32<double,3>();
    // cout<<cuboid<<endl<<endl;
    for(int i = 1;i<cuboid.size(0)+cuboid.size(1)+cuboid.size(2); i++){
        int num_dims = 1+(i+1)/32;
        path_counter<<< dim3(num_dims,num_dims), dim3(32,32)>>>(i, grid, cuboid.size(0), cuboid.size(1), cuboid.size(2));
        cudaDeviceSynchronize();
    }
    return cuboid;
}
// int main(int argc, char** argv){
//     torch::Tensor grid = torch::zeros({7,7,7}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kDouble));
//     int center_x = 3;
//     int center_y = 3;
//     int center_z = 3;
    
    


//     return 0;
// }

