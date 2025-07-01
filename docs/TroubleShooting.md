```
ImportError: /home/papras/miniconda3/envs/paprle3.10/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
(required by /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/_rclpy_pybind11.cpython-310-x86_64-linux-gnu.so)
```
conda install -c conda-forge libstdcxx-ng=12
