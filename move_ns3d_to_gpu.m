function [sem, state] = move_ns3d_to_gpu(sem, state)
sem = move_ns3d_struct_to_gpu(sem);
state = move_ns3d_struct_to_gpu(state);
end
