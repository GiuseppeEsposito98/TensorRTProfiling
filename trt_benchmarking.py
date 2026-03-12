from map_tool_box.modules import Utils
from pathlib import Path
import numpy as np
import torch
import os
import map_tool_box.modules.Benchmark as bm
from map_tool_box.scripts.Pytorch_to_TensorRT.tensorrtConversion.common import load_engine, get_binding_info, allocate_bindings, load_numpy_or_random, load_engine
import pycuda.driver as cuda
import json
import argparse
from tqdm import tqdm
import sys
import pickle


def control():
	total = 0
	nums = [i+1 for i in range(sample_size)]
	for num in nums:
		total *= num


def setup(engine_path):
	# 1) Carica engine e crea contesto
	engine = load_engine(engine_path)

	context = engine.create_execution_context()
	if not context:
		raise RuntimeError("Impossibile creare IExecutionContext.")

	# 4) Alloca buffer H2D/D2H per tutti i binding
	stream = cuda.Stream()

	bindings_ptrs, host_inout, device_inout = allocate_bindings(engine, context, stream)

	return bindings_ptrs, host_inout, device_inout, context, stream


def inference_static(sample_size, bindings_ptrs, host_inout, device_inout, context, stream):
	for name, meta in host_inout.items():
		if meta["is_input"]:
			cuda.memcpy_htod_async(device_inout[name], meta["buffer"], stream)
	for n in range(sample_size):
		# H2D per tutti gli input

		# Inference
		ok = context.execute_v2(bindings_ptrs)
		if not ok:
			raise RuntimeError("execute_v2 ha restituito False.")

	# D2H per tutti gli output
	for name, meta in host_inout.items():
		if not meta["is_input"]:
			cuda.memcpy_dtoh_async(meta["buffer"], device_inout[name], stream)

	stream.synchronize()

def benchmark(bindings_ptrs, host_inout, device_inout, context, stream, n_runs, sample_size):
	for name, meta in host_inout.items():
		if meta["is_input"]:
			if name == "obs":
				meta["buffer"][:] = load_numpy_or_random(obs_npy, meta["shape"], meta["dtype"]).ravel()
			elif name == "vec":
				meta["buffer"][:] = load_numpy_or_random(vec_npy, meta["shape"], meta["dtype"]).ravel()
			else:
				# per eventuali input addizionali
				meta["buffer"][:] = load_numpy_or_random(None, meta["shape"], meta["dtype"]).ravel()

	jetson_json, times = bm.run_benchmark(n_runs, 
										inference_static, 
										{
											'sample_size': sample_size,
											'bindings_ptrs': bindings_ptrs, 
											'host_inout': host_inout, 
											'device_inout': device_inout, 
											'context': context,
											'stream': stream
										})
	return jetson_json

def benchmark_acc(bindings_ptrs, host_inout, device_inout, context, stream, n_runs):
	counter = 0
	gt = list()
	with open('./data/inputs/episodes.p', 'rb') as f:
		episodes = pickle.load(f)
	outs = list()
	for episode in episodes:
		for step in episode:
			obs = step['observations']
			exp_out = step['action_value']
			gt.append(exp_out)

			for name, meta in host_inout.items():
				if meta["is_input"]:
					if name == 'input.1':
						meta["buffer"][:] = obs['img'].reshape(-1)
					elif name == 'onnx::Flatten_1':
						meta["buffer"][:] = obs['vec'].reshape(-1)
					
					
	
		# for name, meta in host_inout.items():
		# 	if meta["is_input"]:
					cuda.memcpy_htod_async(device_inout[name], meta["buffer"], stream)
			
	# for n in range(n_runs):
	# 	# H2D per tutti gli input

	# 	# Inference
			ok = context.execute_v2(bindings_ptrs)
			if not ok:
				raise RuntimeError("execute_v2 ha restituito False.")

			for name, meta in host_inout.items():
				if not meta["is_input"]:
					cuda.memcpy_dtoh_async(meta["buffer"], device_inout[name], stream)
					arr = meta["buffer"].reshape(meta["shape"])
					# print(exp_out == arr)
					print(exp_out)
					print(arr)
					print('===========================')
					# sys.exit()
					outs.append(arr)
			stream.synchronize()

			
	print(counter)
	print(len(gt))

	# for name, meta in host_inout.items():
	# 	if not meta["is_input"]:
	# 		# print(meta['shape'])
	# 		arr = meta["buffer"].reshape(meta["shape"])
	# 		print(arr)

def save_stats(jetson_json, file_path):
	stats = json.loads(jetson_json)
	with open(file_path, 'w') as f:
		json.dump(stats, f, indent=4)

def main():

	ap = argparse.ArgumentParser(description="Benchmarking NN performance")
	ap.add_argument("--root", default = "./ConvertedNNs", help="Root model folder")
	ap.add_argument("--runs", default=100, help="Number of experimental runs")
	ap.add_argument("--samples", default=1000, help="Hardening techniques under test")
	ap.add_argument("--eval_mode", default='energy', help="Evaluation mode")
	args = ap.parse_args()

	sample_size = int(args.samples)
	n_runs = int(args.runs)
	
	root_modules_path = f'{args.root}'

	if args.eval_mode in ['energy', 'acc']:
		for root, dirs, files in os.walk(root_modules_path):
			for file in files:
				if file.endswith('.plan') and 'NH' in root:
					root_module_file_path = os.path.join(root, file)
			
					qnet_bindings_ptrs, qnet_host_inout, qnet_device_inout, qnet_context, qnet_stream = setup(root_module_file_path)

					if args.eval_mode == 'energy':
						obs_npy = None
						vec_npy = None

						device='cuda'
						
						for run_idx in tqdm(range(n_runs), desc = 'Runs'):

								qnet_json = benchmark(qnet_bindings_ptrs, qnet_host_inout, qnet_device_inout, qnet_context, qnet_stream, n_runs, sample_size)

								file_path = os.path.join(root_modules_path, f"{file.split('.')[0]}_{run_idx}.json")
								
								save_stats(qnet_json, file_path)
					elif args.eval_mode == 'acc':
						
						benchmark_acc(qnet_bindings_ptrs, qnet_host_inout, qnet_device_inout, qnet_context, qnet_stream, n_runs)
	else:
		raise NotImplementedError(f'{args.eval_mode} evaluation mode that you are requesting has not been implemented yes')


if __name__ == '__main__':
	main() 
