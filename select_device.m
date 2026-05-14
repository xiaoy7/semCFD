function device = select_device()
if gpuDeviceCount('available') < 1
    device.type = 'cpu';
    device.handle = [];
else
    device.type = 'gpu';
    device.handle = gpuDevice(1);
end
end
