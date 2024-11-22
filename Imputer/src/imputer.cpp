#include "imputer.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



using namespace std;
using namespace torch::indexing;
torch::Tensor visibility_grid(torch::Tensor grid, int center_z, int center_y, int center_x){
    
    CHECK_INPUT(grid);
    grid[center_z][center_y][center_x] = 1;
    
    torch::Tensor out = grid.clone();
    // std::vector<torch::IntArrayRef> flips = {{},{0}, {1}, {2}, {0,1}, {0,2}, {1,2}, {0,1,2}};
    std::vector<std::vector<at::indexing::TensorIndex>> slices = {
        {Slice(center_z, None), Slice(center_y, None), Slice(center_x, None)},
        {Slice(None, center_z+1), Slice(center_y, None), Slice(center_x, None)},
        {Slice(center_z, None), Slice(None, center_y+1), Slice(center_x, None)},
        {Slice(center_z, None), Slice(center_y, None), Slice(None, center_x+1)},
        {Slice(None, center_z+1), Slice(None, center_y+1), Slice(center_x, None)},
        {Slice(None, center_z+1), Slice(center_y, None), Slice(None, center_x+1)},
        {Slice(center_z, None), Slice(None, center_y+1), Slice(None, center_x+1)},
        {Slice(None, center_z+1), Slice(None, center_y+1), Slice(None, center_x+1)},

    };

    std::vector<std::vector<long>> flips = {{},{0}, {1}, {2}, {0,1}, {0,2}, {1,2}, {0,1,2}};

    for(int i=0;i<8;i++){
        torch::Tensor cuboid = grid.index(slices[i]);
        if(i>0){
            cuboid = cuboid.flip(flips[i]);
        }
        cuboid = process_octant(cuboid);
        if(i>0){
            cuboid = cuboid.flip(flips[i]);
        }
        out.index(slices[i]) = cuboid;
        // cout<<grid<<endl<<endl<<endl;
    }
    return out;
}

// int main(){
//     torch::Tensor grid = torch::zeros({33,33,33}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kDouble));
//     torch::Tensor a = visibility_grid(grid, 3,3,3);
//     cout<<a<<endl;
//     return 0;
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("visibility_grid", &visibility_grid, "visibility grid");
}
