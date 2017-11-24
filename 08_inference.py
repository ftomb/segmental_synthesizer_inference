from tensorflow.python.framework import graph_util
from scipy import signal as sg
from scipy.io import wavfile
import tensorflow as tf
import urllib.request
import pyworld as pw
import numpy as np
import pysptk
import pickle
import math
import tgt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def load_TextGrid_titles():
	return [os.path.splitext(fn)[0] for fn in os.listdir('input/') if os.path.splitext(fn)[1] == '.TextGrid']

def load_f0_titles():
	return [os.path.splitext(fn)[0] for fn in os.listdir('input/') if os.path.splitext(fn)[1] == '.f0']

def load_f0(title):
	with open('input/'+title+'.f0') as f:
		return np.array([float(l.strip()) for l in f], dtype=np.float64)

def process_phones(l):
	l_idx = [0]+[0 if l[i] == l[i-1] else 1 for i in range(1, len(l))]
	idxs = [0]+[i for i in range(0, len(l_idx)) if l_idx[i] == 1]+[len(l)]
	slices = [[idxs[i], idxs[i+1]] for i in range(0, len(idxs)-1)]
	l_phs = [l[s[0]:s[1]] for s in slices]
	l_ph_len = [len(i) for i in l_phs]
	l_ph_mid = [i[0] for i in l_phs]
	l_ph_bef = ['sil']+l_ph_mid[:-1]
	l_ph_bef_bef = ['sil']+l_ph_bef[:-1]
	l_ph_aft = l_ph_mid[1:]+['sil']
	l_ph_aft_aft = l_ph_aft[1:]+['sil']
	l_ph_befs = [l_ph_bef[i] for i in range(len(l_ph_len)) for j in range(l_ph_len[i])]
	l_ph_bef_befs = [l_ph_bef_bef[i] for i in range(len(l_ph_len)) for j in range(l_ph_len[i])]
	l_ph_afts = [l_ph_aft[i] for i in range(len(l_ph_len)) for j in range(l_ph_len[i])]
	l_ph_aft_afts = [l_ph_aft_aft[i] for i in range(len(l_ph_len)) for j in range(l_ph_len[i])]
	l_ph_lens = [[l_ph_len[i]] for i in range(len(l_ph_len)) for j in range(l_ph_len[i])]
	l_perc = [[j/len(l_phs[i])] for i in range(len(l_phs)) for j in range(len(l_phs[i]))]
	
	return l, l_ph_befs, l_ph_bef_befs, l_ph_afts, l_ph_aft_afts, l_perc, l_ph_lens

def smooth(a, wl):
	acc = []
	b = np.hsplit(a, len(a[0]))
	for v in b:
		f = v.flatten()
		g = sg.savgol_filter(f, wl, 2)
		acc.append(np.hsplit(g, len(g)))
	return np.concatenate(acc, axis=1)

os.makedirs('input', exist_ok=True)
os.makedirs('output', exist_ok=True)

# Load titles
TextGrid_titles = load_TextGrid_titles()
f0_titles = load_f0_titles()
titles = list(set(TextGrid_titles).intersection(f0_titles))

if titles == []:
	print('No input files found! Please provide .f0 and .TextGrid files!')
	exit()

# Load NN Model 
print('Loading Model...')

with tf.gfile.GFile('build/model/frozen_model', "rb") as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
	tf.import_graph_def(graph_def, input_map=None, return_elements=None, name=None, op_dict=None, producer_op_list=None)

X = graph.get_tensor_by_name('import/X:0')
#seq_len = graph.get_tensor_by_name('import/seq_len:0')
n_frames = graph.get_tensor_by_name('import/n_frames:0')
Y_ = graph.get_tensor_by_name('import/Y_:0')


# Loop over each title
for title in titles:
	print(title)
	print('Preparing input...')
	# Load f0
	f0 = load_f0(title)
	lf0 = np.log2([[float(i)] for i in f0], dtype=np.float64)

	# Generate time points for phone extraction
	ts = [i*0.005 for i in range(len(f0))]

	# Load TextGrid
	tgname = 'input/' + title + '.TextGrid'
	tg = tgt.read_textgrid(tgname)
	phones_tier = tg.get_tier_by_name('phones')

	# Generate phone list
	phone_list = []
	for t in ts:
		try:
			phone_list.append(phones_tier.get_annotations_by_time(t)[0].text)
		except:
			phone_list.append('sil')

	# generate phone vectors
	ph, ph_b, ph_b_b, ph_a, ph_a_a, ph_perc, ph_len = process_phones(phone_list)

	with open('phone_dictionary.dict', "rb") as f:
		ph_dict = pickle.load(f)

	# Convert phone vectors to hot vectors
	hot_ph = np.array([ph_dict[i] for i in ph])
	hot_ph_b = np.array([ph_dict[i] for i in ph_b])
	hot_ph_b_b = np.array([ph_dict[i] for i in ph_b_b])
	hot_ph_a = np.array([ph_dict[i] for i in ph_a])
	hot_ph_a_a = np.array([ph_dict[i] for i in ph_a_a])

	# Concatenate all the input vectors
	input_vector = np.concatenate((hot_ph, hot_ph_b, hot_ph_b_b, hot_ph_a, hot_ph_a_a, ph_perc, ph_len, lf0), axis=1)

	with open('input_mean_std.pickle' , "rb") as f: 
		input_mean, input_std = pickle.load(f)

	# Normalize input vector
	print('Normalizing...')
	input_vector -= input_mean
	input_vector /= input_std+np.finfo(float).eps

	# Inference
	print('Doing inference...')
	length = len(input_vector)
	with tf.Session(graph=graph) as sess:
		#output_vector = sess.run([Y_], {X:input_vector, seq_len:[length], n_frames:length})
		output_vector = sess.run([Y_], {X:input_vector, n_frames:length})
		#output_vector = sess.run([Y_], {X:input_vector})
	prediction = output_vector[0]
	#prediction = smooth(prediction, 5)


	print('Denormalizing...')
	with open('output_mean_std' + '.pickle', "rb") as g: 
		output_mean, output_std = pickle.load(g)

	prediction *= output_std
	prediction += output_mean

	fs = 48000

	vuv = np.array(prediction[:,0:1], dtype=np.float64)
	vuv = np.array([[1] if i[0] > 0.5 else [0] for i in vuv])
	bap = np.array(prediction[:,1:3], dtype=np.float64)
	mgc = np.array(prediction[:,3:], dtype=np.float64)
	f0 = load_f0(title)

	bap = bap*vuv
	f0 = f0*vuv.flatten()

	sp = pysptk.mc2sp(mgc, fftlen=1024, alpha=pysptk.util.mcepalpha(fs))
	ap = pysptk.mc2sp(bap, fftlen=1024, alpha=pysptk.util.mcepalpha(fs))

	y = pw.synthesize(f0, sp, ap, fs)
	wavfile.write('output/' + title + '.wav', fs, np.array(y, dtype=np.int16))


