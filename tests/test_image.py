import torch
from mc_cnn.weights import get_weights
from mc_cnn.model.mc_cnn_fast import FastMcCnn
import rasterio
from mc_cnn.run import run_mc_cnn_fast, run_mc_cnn_accurate
from pandora import disparity
import xarray as xr
import numpy as np

if __name__ == '__main__':

    # Read images (only grayscale images are accepted )
    left_image = rasterio.open('/work/data/i18R/REMAP/TRAIN/EVT/depth_data2_2023_03_06/depth_data2_1loukongdi/img'
                               '/left/1001677897568288285.png').read(1)
    right_image = rasterio.open('/work/data/i18R/REMAP/TRAIN/EVT/depth_data2_2023_03_06/depth_data2_1loukongdi/img'
                                '/right/1001677897568288285.png').read(1)
    disparity_min = 0
    disparity_max = 4
    print("left_image.shape: ", left_image.shape)
    # Path to the pretrained network
    mccnn_fast_model_path = str(get_weights(arch="fast", training_dataset="middlebury")) # Or custom weights filepath
    mccnn_accurate_model_path = str(get_weights(arch="accurate", training_dataset="middlebury"))
    print("sssssssss")
    # Cost volume using mccnn fast similarity measure
    cost_volume_fast = run_mc_cnn_fast(left_image, right_image, disparity_min, disparity_max, mccnn_fast_model_path)

    # Cost volume using mccnn accurate similarity measure
    cost_volume_accurate = run_mc_cnn_accurate(left_image, right_image
                                               , disparity_min, disparity_max, mccnn_accurate_model_path)
    # print("cost_volume_fast: ", cost_volume_fast)
    print("cost_volume_fast.shape: ", cost_volume_fast.shape)

    disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
    cv = xr.Dataset({"cost_volume" : (["row", "col", "disp"], cost_volume_fast)},
                    coords={"row":np.arange(cost_volume_fast.shape[0])
                        , 'col':np.arange(cost_volume_fast.shape[1])
                        , "disp":np.arange(cost_volume_fast.shape[2])})
    cv.attrs["type_measure"] = "min"
    cv.attrs["measure"] = "sad"
    cv.attrs["subpixel"] = 1
    cv.attrs["offset_row_col"] = 0
    cv.attrs["window_size"] = 5
    cv.attrs["cmax"] = 0
    cv.attrs["band_correl"] = None
    cv.attrs["crs"] = None
    print(cv.coords["row"])
    print("===========cv: ", cv)
    result = disparity_.to_disp(cv, left_image, right_image)
    print("result: ", result)

    print("end!")
