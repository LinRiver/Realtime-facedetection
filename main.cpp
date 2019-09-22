#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

String ageList[] = {"(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"};
String genderList[] = {"Male", "Female"};

void recognize_face(Mat& face, Net net, vector<float> &fv);
float compare(vector<float> &fv1, vector<float> &fv2);

int main() {
    string facenet_model = "/home/riverlin/Pretrained_Model/facenet_models/openface.nn4.small2.v1.t7";
    string pb_model = "/home/riverlin/SLAM_Lib/opencv-3.4.3/samples/dnn/face_detector/opencv_face_detector_uint8.pb";
    string pb_txt = "/home/riverlin/SLAM_Lib/opencv-3.4.3/samples/dnn/face_detector/opencv_face_detector.pbtxt";

    string age_model = "/home/riverlin/Pretrained_Model/cnn_age_gender_models/age_net.caffemodel";
    string gender_model = "/home/riverlin/Pretrained_Model/cnn_age_gender_models/gender_net.caffemodel";
    string age_protxt = "/home/riverlin/Pretrained_Model/cnn_age_gender_models/age_deploy.prototxt";
    string gender_protxt = "/home/riverlin/Pretrained_Model/cnn_age_gender_models/gender_deploy.prototxt";

    // load pretrained model
    Net face_net = readNetFromTorch(facenet_model);
    Net net = readNetFromTensorflow(pb_model, pb_txt);
    Net age_net = readNetFromCaffe(age_protxt, age_model);
    Net gender_net = readNetFromCaffe(gender_protxt, gender_model);

    // set up the backend computation
    face_net.setPreferableBackend(DNN_BACKEND_OPENCV);
    face_net.setPreferableTarget(DNN_TARGET_CPU);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    age_net.setPreferableBackend(DNN_BACKEND_OPENCV);
    age_net.setPreferableTarget(DNN_TARGET_CPU);
    gender_net.setPreferableBackend(DNN_BACKEND_OPENCV);
    gender_net.setPreferableTarget(DNN_TARGET_CPU);

    // load face data for face_net training
    vector<vector<float>> face_data;
    vector<cv::String> labels;
    vector<cv::String> faces;
    cv::glob("/home/riverlin/CLionProjects/Realtime_facedetection/riverlin", faces);
    for(auto fn:faces){
        vector<float> fv;
        Mat sample = imread(fn);
        recognize_face(sample, face_net, fv);
        face_data.push_back(fv);
        labels.push_back("riverlin");
    }
    faces.clear();

    // start real-time videocapture for face recognition, age and gender detection
    VideoCapture capture(0);
    Mat frame;
    while(true){
        bool ret = capture.read(frame);
        if(!ret) break;
        flip(frame, frame, 1);

        Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);
        net.setInput(blob, "data");

        Mat detection = net.forward("detection_out");
        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        float confidence_threshold = 0.5;

        int padding = 15;
        for(int i = 0; i < detectionMat.rows; i++){
            float* curr_row = detectionMat.ptr<float>(i);
            int image_id = (int)(*curr_row++);
            size_t objIndex = (size_t)(*curr_row++);
            float score = *curr_row++;
            if(score>confidence_threshold){
                float tl_x = (*curr_row++)*frame.cols;
                float tl_y = (*curr_row++)*frame.rows;
                float br_x = (*curr_row++)*frame.cols;
                float br_y = (*curr_row++)*frame.rows;
                Rect box((int)tl_x, (int)tl_y, (int)(br_x-tl_x), (int)(br_y-tl_y));

                Rect roi;
                roi.x = max(0, box.x - padding);
                roi.y = max(0, box.y - padding);
                roi.width = min(box.width + padding, frame.cols - 1);
                roi.height = min(box.height + padding, frame.rows - 1);

                Mat face = frame(roi);
                Mat face_blob = blobFromImage(face, 1.0, Size(227, 227),
                        Scalar(78.4263377603, 87.7689143744, 114.895847746), false, false);
                age_net.setInput(face_blob);
                gender_net.setInput(face_blob);
                Mat ageProbs = age_net.forward();
                Mat genderProbs = gender_net.forward();

                Mat prob_age = ageProbs.reshape(1, 1);
                Point classNum;
                double classProb;
                minMaxLoc(prob_age, 0, &classProb, 0, &classNum);
                int classIdx = classNum.x;
                String age = ageList[classIdx];

                Mat prob_gender = genderProbs.reshape(1, 1);
                minMaxLoc(prob_gender, 0, &classProb, 0, &classNum);
                classIdx = classNum.x;
                String gender = genderList[classIdx];

                rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
                putText(frame, format("age:%s, gender:%s", age.c_str(), gender.c_str()), box.tl(),
                        FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 0, 0), 3);

// TODO: face_net can work in the ROI area of age and gender detection

//                Rect roi_recognize;
//                roi.x = max(0, box.x);
//                roi.y = max(0, box.y);
//                roi.width = min(box.width, frame.cols - 1);
//                roi.height = min(box.height, frame.rows - 1);
//                Mat face_recognize = frame(roi_recognize);
//
//                vector<float> curr_fv;
//                recognize_face(face_recognize, face_net, curr_fv);
//
//                // calculate similary
//                float minDist = 10;
//                int index = -1;
//                for(int i = 0; face_data.size(); i++){
//                    float dist = compare(face_data[i], curr_fv);
//
//                    if(minDist > dist){
//                        minDist = dist;
//                        index = i;
//                    }
//                }
//
//                if (minDist < 0.30 && index >= 0) {
//                    putText(frame, format("%s", labels[index].c_str()), Point(roi_recognize.x, roi_recognize.y-10),
//                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8);
//                }
//                rectangle(frame, box, Scalar(255, 0, 255), 1, 8, 0);

            }
        }

        vector<double> layersTimings;
        double freq = getTickFrequency() / 1000.0;
        double time = net.getPerfProfile(layersTimings) / freq;
        ostringstream ss;
        ss << "FPS: " << 1000 / time << " ; time : " << time << "ms";

        putText(frame, ss.str(), Point(20, 20), FONT_HERSHEY_PLAIN,
                1.0, Scalar(255, 0, 0), 2, 8);
        imshow("face-funtionaldetection-demo", frame);
        char c = waitKey(1);
        if(c == 27) break;
    }

    capture.release();
    waitKey(0);
    return 0;
}

// face image recognition with vector operation for face_net
void recognize_face(Mat& face, Net net, vector<float> &fv){
    Mat blob = blobFromImage(face, 1/255.0, Size(96, 96),
            Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    Mat probMat = net.forward();
    Mat vec = probMat.reshape(1, 1);

    for(int i = 0; i < vec.cols; i++){
        fv.push_back(vec.at<float>(0, i));
    }
}

// calculate the similary with cos function for face_net
float compare(vector<float> &fv1, vector<float> &fv2){
    float dot = 0;
    float sum2 = 0;
    float sum3 = 0;
    for(int i = 0; i < fv1.size(); i++){
        dot += fv1[i]*fv2[i];
        sum2 += pow(fv1[i], 2);
        sum3 += pow(fv2[i], 2);
    }
    float norm = sqrt(sum2)*sqrt(sum3);
    float similary = dot / norm;
    float dist = acos(similary) / CV_PI;
    return dist;
}