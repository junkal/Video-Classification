from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import sys
import os

def load_classifier_model(model_path):
    model = None
    try:
        print("[INFO] Loading model from {}".format(model_path))
        model = load_model(model_path)
    except Exception as error:
        print("[Error] {}".format(error))

    return model

def load_label(label_bin):
    lb = None
    try:
        print("[INFO] Loading label binarizer from {}".format(label_bin))
        lb = pickle.loads(open(label_bin, "rb").read())
    except Exception as error:
        print("[Error] {}".format(error))

    return lb


def analyse_video(model, input_video, labels, output_path, q_size = 64):
    # initialize the image mean for mean subtraction along with the
    # predictions queue
    mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
    Q = deque(maxlen = q_size)

    if os.path.isfile(input_video):
        vs = cv2.VideoCapture(input_video)
    else:
        print("[Error] Unable to locate file {}".format(input_video))
        return

    writer = None
    (W, H) = (None, None)

    output_filename = os.path.splitext(os.path.basename(input_video))[0] + "_output.avi"
    output_video = os.path.join(output_path, output_filename)

    while True:
    	# read the next frame from the file
    	(grabbed, frame) = vs.read()
    	# if the frame was not grabbed, then we have reached the end
    	# of the stream
    	if not grabbed:
    		break
    	# if the frame dimensions are empty, grab them
    	if W is None or H is None:
    		(H, W) = frame.shape[:2]
            # clone the output frame, then convert it from BGR to RGB
    	# ordering, resize the frame to a fixed 224x224, and then
    	# perform mean subtraction
    	output = frame.copy()
    	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    	frame = cv2.resize(frame, (224, 224)).astype("float32")
    	frame -= mean
        # make predictions on the frame and then update the predictions queue
    	preds = model.predict(np.expand_dims(frame, axis=0))[0]
    	Q.append(preds)

    	# perform prediction averaging over the current history of
    	# previous predictions
    	results = np.array(Q).mean(axis=0)
    	i = np.argmax(results)
    	label = labels.classes_[i]

        # write the activity on the output frame
    	text = "activity: {}".format(label)
    	cv2.putText(output, text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    	if writer is None:
    		# initialize our video writer
    		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    		writer = cv2.VideoWriter(output_video, fourcc, 30, (W, H), True)

        # write the output frame to disk
    	writer.write(output)

    # release the file pointers
    print("[INFO] Analysing video file {} completed".format(input_video))

    writer.release()
    vs.release()

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type = str, required=True, help="path to trained serialized model")
    parser.add_argument("-l", "--label", type = str, required=True, help="path to  label binarizer")
    parser.add_argument("-i", "--input", type = str, required=True, help="path to our input video")
    parser.add_argument("-o", "--output", type = str, required=True, help="path to our output video")
    parser.add_argument("-s", "--size", type= int, default=128, help="size of queue for averaging")

    return parser

def main(args=None):
    if os.name == 'posix':
        os.system('clear')
    else:
      os.system('cls') # for windows platfrom

    parser = parse_opt()
    args = parser.parse_args(args)

    print("[INFO] Input model path : {}".format(args.model))
    print("[INFO] Input label      : {}".format(args.label))
    print("[INFO] Input video      : {}".format(args.input))
    print("[INFO] Output video     : {}".format(args.output))
    print("[INFO] Average Q size   : {}".format(args.size))

    if not os.path.exists(args.model):
        print("[ERROR] Model file is not available at {}".format(args.model))
        sys.exit(1)

    if not os.path.exists(args.label):
        print("[ERROR] Label file is not available at {}".format(args.label))
        sys.exit(1)

    if not os.path.exists(args.input):
        print("[ERROR] Input data file is not available at {}".format(args.input))
        sys.exit(1)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model = load_classifier_model(args.model)
    if  model == None:
        sys.exit(1)

    lb = load_label(args.label)
    if lb == None:
        sys.exit(1)

    if os.path.isfile(args.input):
        analyse_video(model, args.input, lb, args.output, q_size = args.size)
    else:
        for file in os.listdir(args.input):
            input_file = os.path.join(args.input, file)
            analyse_video(model, input_file, lb, args.output, q_size = args.size)

if __name__ == '__main__':
    main()
