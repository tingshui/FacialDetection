/*Qianying Lin computer vision project 2*/

#include "opencv2/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include "opencv2/dirent.h"

using namespace std;
using namespace cv;


/*
The cascade classifiers that come with opencv are kept in the
following folder: bulid/etc/haarscascades
Set OPENCV_ROOT to the location of opencv in your system
*/
string OPENCV_ROOT = "C:/opencv/";
string cascades = OPENCV_ROOT + "build/etc/haarcascades/";
string FACES_CASCADE_NAME = cascades + "haarcascade_frontalface_alt.xml";
string EYES_CASCADE_NAME = cascades + "haarcascade_eye.xml";


void drawEllipse(Mat frame, const Rect rect, int r, int g, int b) {
	int width2 = rect.width / 2;
	int height2 = rect.height / 2;
	Point center(rect.x + width2, rect.y + height2);
	ellipse(frame, center, Size(width2, height2), 0, 0, 360,
		Scalar(r, g, b), 2, 8, 0);
}


// Copy specific region for detecting eyes, change from the histogram data to a muti-dimensional array
Mat StandardEye(int cols, int rows, Mat OrigEye, Point Center) {
	//  Mat Eye(rows,cols,CV_BGR2GRAY);
	Mat Eye(rows, cols, CV_HIST_ARRAY);
	int StartRow = Center.y - (rows / 2);
	int StartCol = Center.x - (cols / 2);
	for (int i = 0; i< rows; i++)
		for (int j = 0; j<cols; j++) {
			int OrigRow = StartRow + i;
			int OrigCol = StartCol + j;
			Eye.at<uchar>(i, j) = OrigEye.at<uchar>(OrigRow, OrigCol);
		}
	return Eye;
}


int detectWink(Mat frame, Mat frame_gray, Point location, Mat ROI, CascadeClassifier cascade, bool finalStep) {
	// frame,ctr are only used for drawing the detected eyes

	//The value below is the center of the face
	int face_center_x = ROI.cols / 2;
	int face_center_y = ROI.rows / 2;

	int num_eye_left = 0;          //number of eyes on the left upper face
	int num_eye_right = 0;         //number of eyes on the right upper face
	int num_eye_lower = 0;         //number of eyes on the lower face
	int eyecount = 0;

	// Get all possible detected eyes
	vector<Rect> eyes;
	cascade.detectMultiScale(ROI, eyes, 1.1, 3, 0, Size(20, 20));
	int neyes = (int)eyes.size();

	// initialization, overwrite the eyes array with -1
	int rightEyes[100];
	memset(rightEyes, -1, sizeof rightEyes);
	int leftEyes[100];
	memset(leftEyes, -1, sizeof leftEyes);
	int lowerEyes[100];
	memset(lowerEyes, -1, sizeof lowerEyes);

	// check the locations of eyes
	for (int i = 0; i < neyes; i++) {
		Rect eyes_i = eyes[i];

		//Get the center of eye
		int width2 = eyes_i.width / 2;
		int height2 = eyes_i.height / 2;
		Point center(eyes_i.x + width2, eyes_i.y + height2);

		//If the center of eye is below the center of face
		if (center.y > face_center_y) {
			lowerEyes[num_eye_lower] = i;
			num_eye_lower++;
		}
		// the eye is in the upper face
		else {
			//the eye is on the left of the upper face
			if (center.x < face_center_x) {
				leftEyes[num_eye_left] = i;
				num_eye_left++;
			}
			//the eye is on the right of the upper face
			else {
				rightEyes[num_eye_right] = i;
				num_eye_right++;
			}
		}
	}

	//if there are 2 or more eyes on the same side, keep the larger one
	// left side
	int maxLeftEyeSize = 0;
	// -1 means not exist
	int maxLeftEyeIndex = -1;
	for (int i = 0; i < num_eye_left; i++) {
		int n = leftEyes[i];
		int size = MAX(eyes[n].width, eyes[n].height);
		if (size > maxLeftEyeSize) {
			maxLeftEyeSize = size;
			maxLeftEyeIndex = n;
		}
	}
	// right side
	int maxRightEyeSize = 0;
	int maxRightEyeIndex = -1;
	for (int i = 0; i < num_eye_right; i++) {
		int n = rightEyes[i];
		int size = MAX(eyes[n].width, eyes[n].height);
		if (size > maxRightEyeSize) {
			maxRightEyeSize = size;
			maxRightEyeIndex = n;
		}
	}

	// If detect 2 eyes in final step (one left and one right), compare the two eyes to check whether winked
	if (maxRightEyeIndex != -1 && maxLeftEyeIndex != -1 && finalStep) {
		Rect Reye = eyes[maxRightEyeIndex];
		Rect Leye = eyes[maxLeftEyeIndex];
		Mat ReyeROI = frame(Reye);
		Mat LeyeROI = frame(Leye);

		//find the maximum rows & cols 
		int cols = MAX(ReyeROI.cols, LeyeROI.cols);
		int rows = MAX(ReyeROI.rows, LeyeROI.rows);

		Point ReyeCenter, LeyeCenter;
		ReyeCenter.x = Reye.x + (ReyeROI.cols / 2);
		ReyeCenter.y = Reye.y + (ReyeROI.rows / 2);

		LeyeCenter.x = Leye.x + (LeyeROI.cols / 2);
		LeyeCenter.y = Leye.y + (LeyeROI.rows / 2);

		ReyeROI = StandardEye(cols, rows, frame, ReyeCenter);
		LeyeROI = StandardEye(cols, rows, frame, LeyeCenter);

		void cvFlip(const CvArr* LeyeROI, CvArr* dst = NULL, int flip_mode = 0);

		equalizeHist(ReyeROI, ReyeROI);
		equalizeHist(LeyeROI, LeyeROI);

		int histSize = 256;
		float range[] = { 0, 256 };
		const float* histRange = { range };
		bool uniform = true; bool accumulate = false;
		Mat hist1;
		Mat hist2;
		calcHist(&ReyeROI, 1, 0, Mat(), hist1, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&LeyeROI, 1, 0, Mat(), hist2, 1, &histSize, &histRange, uniform, accumulate);

		// Compare the two eyes
		double result = compareHist(hist1, hist2, CV_COMP_BHATTACHARYYA);
		// if the similarity is high, the two eyes are both open or winked, otherwise it is winked face
		if (result > 0.9) {
			eyecount = 1;
		}
	}

	//Draw eclipse for left/right eyes.

	if (maxLeftEyeIndex != -1 && finalStep) {
		Rect eyes_i = eyes[maxLeftEyeIndex];
		drawEllipse(frame, eyes_i + location, 255, 255, 0);
	}

	if (maxRightEyeIndex != -1 && finalStep) {
		Rect eyes_i = eyes[maxRightEyeIndex];
		drawEllipse(frame, eyes_i + location, 255, 255, 0);
	}

	if (eyecount != 1) {
		if (maxRightEyeIndex >= 0)
			eyecount++;
		if (maxLeftEyeIndex >= 0)
			eyecount++;
	}

	return eyecount;
}


int winkedface(Mat frame, Mat frame_gray, CascadeClassifier cascade_face, CascadeClassifier cascade_eyes, bool finalStep) {
	vector<Rect> faces;

	// detect possible faces
	cascade_face.detectMultiScale(frame_gray, faces,
		1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	int detected = 0;

	int nfaces = (int)faces.size();
	bool Winkedfaces[100];
	memset(Winkedfaces, false, sizeof Winkedfaces);

	for (int i = 0; i < nfaces; i++) {
		Rect face = faces[i];
		Mat faceROI = frame_gray(face);
		int NumberOfEye = detectWink(frame, frame_gray, Point(face.x, face.y), faceROI, cascade_eyes, finalStep);
		if (NumberOfEye == 1) {
			Winkedfaces[i] = true;
			detected++;
		}
	}

	//return already detected face number and draw the circle
	if (detected == nfaces || finalStep) {
		for (int i = 0; i < nfaces; i++) {
			Rect face = faces[i];
			if (!finalStep) {
				finalStep = true;
				Mat faceROI = frame_gray(face);
				detectWink(frame, frame_gray, Point(face.x, face.y), faceROI, cascade_eyes, finalStep);
				finalStep = false;
			}
			drawEllipse(frame, face, 255, 0, 255);
			if (Winkedfaces[i])
				drawEllipse(frame, face, 0, 255, 0);
		}
		/*
		string windowName;

		if (!windowName.empty()) destroyWindow(windowName);
		windowName = "meditate";
		namedWindow(windowName.c_str(), CV_WINDOW_AUTOSIZE);
		imshow(windowName.c_str(), frame_gray);
		*/
		return detected;
	}
	else
		return 0;
}


int detect2(Mat frame, CascadeClassifier cascade_face, CascadeClassifier cascade_eyes) {
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	int detected = 0;
	bool finalStep = false;
	//detect winked face
	detected = winkedface(frame, frame_gray, cascade_face, cascade_eyes, finalStep);
	//if not find wink, do Hist equalize and detect again;
	if (detected == 0) {
		equalizeHist(frame_gray, frame_gray);
		detected = winkedface(frame, frame_gray, cascade_face, cascade_eyes, finalStep);
		//if not find wink, do the median filter and detect again;
		if (detected == 0) {
			medianBlur(frame_gray, frame_gray, 5);
			// finalStep = true;
			detected = winkedface(frame, frame_gray, cascade_face, cascade_eyes, finalStep);
			if (detected == 0) {
				blur(frame_gray, frame_gray, Size(5, 5), Point(-1, -1));
				finalStep = true;
				detected = winkedface(frame, frame_gray, cascade_face, cascade_eyes, finalStep);
			}
		}
	}

	return detected;
}



bool  getimage(const string folder, DIR *dir, Mat &img) {
	struct dirent *entry = readdir(dir);
	if (entry == NULL) return(false);
	char *name = entry->d_name;
	string dname = folder + name;
	img = imread(dname.c_str(), CV_LOAD_IMAGE_UNCHANGED);
	return(true);
}

int runonFolder(const CascadeClassifier cascade1,
	const CascadeClassifier cascade2,
	string folder) {
	if (folder.at(folder.length() - 1) != '/') folder += '/';
	DIR *dir = opendir(folder.c_str());
	if (dir == NULL) {
		cerr << "Can't open folder " << folder << endl;
		exit(1);
	}
	bool finish = false;
	string windowName;
	struct dirent *entry;
	int detections = 0;
	while (!finish && (entry = readdir(dir)) != NULL) {
		char *name = entry->d_name;
		string dname = folder + name;
		Mat img = imread(dname.c_str(), CV_LOAD_IMAGE_UNCHANGED);
		if (!img.empty()) {
			int d = detect2(img, cascade1, cascade2);
			cerr << d << " detections" << endl;
			detections += d;
			if (!windowName.empty()) destroyWindow(windowName);
			windowName = name;
			namedWindow(windowName.c_str(), CV_WINDOW_AUTOSIZE);
			imshow(windowName.c_str(), img);
			int key = waitKey(0); // Wait for a keystroke
			switch (key) {
			case 27: // <Esc>
				finish = true; break;
			case 13: // <Enter>
				break;
			default:
				break;
			}
		} // if image is available
	}
	closedir(dir);
	return(detections);
}

void runonVideo(const CascadeClassifier cascade1,
	const CascadeClassifier cascade2) {
	VideoCapture videocapture(0);
	if (!videocapture.isOpened()) {
		cerr << "Can't open default video camera" << endl;
		exit(1);
	}
	string windowName = "Live Video";
	namedWindow("video", CV_WINDOW_AUTOSIZE);
	Mat frame;
	bool finish = false;
	while (!finish) {
		if (!videocapture.read(frame)) {
			cout << "Can't capture frame" << endl;
			break;
		}
		detect2(frame, cascade1, cascade2);
		imshow("video", frame);
		if (cvWaitKey(30) >= 0) finish = true;
	}
}

int main(int argc, char** argv) {
	if (argc != 1 && argc != 2) {
		cerr << argv[0] << ": "
			<< "got " << argc - 1
			<< " arguments. Expecting 0 or 1 : [image-folder]"
			<< endl;
		return(-1);
	}

	string foldername = (argc == 1) ? "" : argv[1];
	CascadeClassifier faces_cascade, eyes_cascade;
	if (
		!faces_cascade.load(FACES_CASCADE_NAME)
		|| !eyes_cascade.load(EYES_CASCADE_NAME)) {
		cerr << FACES_CASCADE_NAME << " or " << EYES_CASCADE_NAME
			<< " are not in a proper cascade format" << endl;
		return(-1);
	}

	int detections = 0;

	if (argc == 2) {
		//        cout<< "run argc == 2";
		//        cout<<"\n"<<foldername<<"\n";
		detections = runonFolder(faces_cascade, eyes_cascade, foldername);
		cout << "Total of " << detections << " detections" << endl;
	}
	else{
		runonVideo(faces_cascade, eyes_cascade);
	}

	return(0);
}