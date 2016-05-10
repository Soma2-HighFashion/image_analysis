import sys
sys.path.append('./model')

import argparse
import input_data
from alexnet_mnist import AlexNet

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-learning_rate', action='store', dest='learning_rate', type=float, 
			help='Learning Rate')	
	parser.add_argument('-num_iters', action='store', dest='num_iters', type=int, 
			help='Training Iteration Count')
	parser.add_argument('-batch_size', action='store', dest='batch_size', type=int, 
			help='Batch Size => mini-batch')
	parser.add_argument('-reg', action='store', dest='reg', type=float,
			help='Regulizer')
	parser.add_argument('-dropout', action='store', dest='dropout', type=float,
			help='Dropout Ratio')

	config = parser.parse_args()

	print "--------------- Config Description ---------------"
	print " -  Learning Rate : ", config.learning_rate
	print " -  Num Iterations : ", config.num_iters
	print " -  Batch Size : ", config.batch_size
	print " -  Regulizer : ", config.reg

	mnist = input_data.read_data_sets("./", one_hot=True)

	geometry = [28, 28]
	num_classes = 10

	model = AlexNet(geometry, num_classes)
	model.train(mnist, learning_rate=config.learning_rate, num_iters=config.num_iters, 
			batch_size=config.batch_size, dropout_prob=config.dropout, verbose=True)

	model.predict(mnist.test.images, mnist.test.labels)
