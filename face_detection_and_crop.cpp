// Revision history:
// [20151119_Curtis] 1. normalize and crop face images for gender classfication
//                   2. use uniform LBP and SVM to train gender classfier
// [20151120_Curtis] show menu for user to select one of the following three functions: a). face detection and cropping, b). training, and c). prediction 
// [20151123_Curtis] add the 4th function to do gender classification for video input
// [20151209_Curtis] remove the 3rd function which predicts gender from a list of files
// [20151216_Curtis] add facial landmark tracking to improve the stability of face detection

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include <opencv/cv.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <dlib/image_transforms.h>
#include "classification.h"
#include "FaceProcessing.h"
#include "FlyCaptureWrapper.h"

//#include "OpenNiWrapper.h"
//Linux
#include <time.h>

//qt
#include <QApplication>
#include <mainwindow.h>
//#define USE_HISTO_EQUAL

int main(int argc, char** argv)
{  
    //FeatureSelectionUseBoost();
    //return 0;

    // show menu
    printf("--------------------------------------\n");
    printf(" gender classification (video)\n");
    printf("--------------------------------------\n");
    //char key = getchar();
    // ---------------------------
    // face detection and cropping
    // ---------------------------

    //DELETED

    // --------------------------
    // gender classifier training
    // --------------------------

    //DELETED

    // -----------------------------
    // gender classification (video)
    // -----------------------------

    // load SVM model
    //[20160104_Sylar]
    //CLbpSvm lbpSvm("./Test/svm.model", "./Test/MinMax.csv");

    //CLbpSvm lbpSvm("./Test/SvmBoostMScale.model", "./Test/MinMax.csv");

    // use camera, ASUS xtion, or video as input source
    //cv::VideoCapture cap(0);
    //COpenNiWrapper openNiWrapper;
    QApplication a(argc, argv);

    MainWindow w;
    cv::gpu::setDevice(0);
    printf("Setting GPU...\n");

    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " deploy.prototxt network.caffemodel"
                  << " mean.binaryproto labels.txt " << std::endl;
        return 1;
    }
    ::google::InitGoogleLogging(argv[0]);
    string model_file   = argv[1];
    string trained_file = argv[2];
    string mean_file    = argv[3];
    string label_file   = argv[4];
    Classifier classifier(model_file, trained_file, mean_file, label_file);
    //cv::VideoCapture cap("/home/sylar/gender_classification/Backstreet.mp4");
    //cv::VideoCapture cap(0);
    //if (!cap.isOpened()) return -1;

    // load the cascades
    /*CFaceProcessing fp("D:/Software/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml",
      "D:/Software/opencv/sources/data/haarcascades/haarcascade_mcs_nose.xml",
      "D:/Software/opencv/sources/data/haarcascades/haarcascade_mcs_mouth.xml",
      "D:/Vision_Project/shape_predictor_68_face_landmarks.dat");*/

    //[20160104_Sylar]
    CFaceProcessing fp("/home/sylar/gender_classification/xml/lbpcascade_frontalface.xml",
                       "/home/sylar/gender_classification/xml/haarcascade_mcs_nose.xml",
                       "/home/sylar/gender_classification/xml/haarcascade_mcs_mouth.xml",
                       "/home/sylar/gender_classification/xml/shape_predictor_68_face_landmarks.dat");

    // main loop
    cv::Mat img;
    bool showLandmark = false;
    bool showCroppedFaceImg = false;
    cv::Mat grayFrame;
    cv::Mat grayFramePrev;
    std::vector<std::vector<cv::Point> > fLandmarksPrev;
    std::vector<std::vector<cv::Point> > fLandmarks;
    std::vector<unsigned char> faceStatusPrev;
    std::vector<float> accGenderConfidencePrev;
    float totalCount = 0;
    float falseCount = 0;
    std::vector<cv::Mat> prevCropped;
    bool enable_gpu = false;
    cv::TickMeter timer;
    timer.start();
    //QT
    w.show();
    FlyCaptureWrapper fly;
    if(!fly.flyCaptureCheckStatus())
        return -1;
    while (1)
    {
        //openNiWrapper.GetDepthColorRaw();
        //openNiWrapper.ConvertDepthColorRawToImage(cv::Mat(), img);
        //cap >> img;
        img = fly.flyCaptureGetFrame();
        if (img.empty()) break;
        cv::resize(img, img, cv::Size(1280, 720));
        // (optional) backup original image for offline debug
        cv::Mat originImg(img.size(), img.type());
        img.copyTo(originImg);

        // time calculation
        cv::TickMeter tm;
        tm.start();

        // -----------------------------------
        // face detection
        // -----------------------------------
        std::vector<cv::Rect> faces;
        // -----------------
        // consume most time
        int faceNum ;
        // Press g to change mode
        if(enable_gpu){
            faceNum = fp.FaceDetection_GPU(img);
            cv::putText(img, "GPU", cv::Point(30, 100), cv::FONT_HERSHEY_COMPLEX, 1, CV_RGB(255, 255, 255));
        }
        else{
            faceNum = fp.FaceDetection(img);
            cv::putText(img, "CPU", cv::Point(30, 100), cv::FONT_HERSHEY_COMPLEX, 1, CV_RGB(255, 255, 255));
        }
        // -----------------
        std::vector<cv::Mat> croppedImgs;
        if (faceNum > 0)
        {
            faces = fp.GetFaces();

            // normalize the face image with landmark
            std::vector<cv::Mat> normalizedImg;
            //fp.AlignFaces2D(normalizedImg, originImg);
            // ----------------------------------------
            // crop faces and do histogram equalization
            // ----------------------------------------
            croppedImgs.resize(faceNum);
            for (int i = 0; i < faceNum; i++)
            {
                // ------------------------------
                // Sylar 20160308 to use RGBscale
                // ------------------------------
                int x = faces[i].x - (faces[i].width / 4);
                int y = faces[i].y - (faces[i].height / 4);
                if (x < 0)
                    x = 0;
                if (y < 0)
                    y = 0;
                int w = faces[i].width + (faces[i].width / 2) ;
                int h = faces[i].height + (faces[i].height / 2);
                if(w + x > originImg.cols)
                    w = originImg.cols - x ;
                if(h + y > originImg.rows)
                    h = originImg.rows - y ;
                croppedImgs[i] = originImg(cv::Rect(x, y, w, h)).clone();
                //cv::resize(croppedImgs[i], croppedImgs[i], cv::Size(227, 227));
                // -------------------------------
                // Sylar 20160308 to use grayscale
                // -------------------------------
                //normalizedImg[i].copyTo(croppedImgs[i]);
#ifdef USE_HISTO_EQUAL
                //cv::equalizeHist(croppedImgs[i], croppedImgs[i]);
#endif
            }
            // ---------------------------------
            // extraction landmarks on each face
            // ---------------------------------
            fLandmarks.resize(faceNum);
            for (int i = 0; i < faceNum; i++)
            {
                // fLandmarks[i] = fp.GetLandmarks(i);
            }
        }
        // (debug) show no face
        //else
        //{
        //   printf("Detect no face\n\n");
        //}

        // ----------------------
        // track facial landmarks
        // ----------------------
        /*
         grayFrame = fp.GetGrayImages();
         std::vector<std::pair<int, int> > trackFromTo;
         
         if (grayFramePrev.empty() == false && fLandmarksPrev.size() != 0) // do tracking when the current frame is not the first one
         {
            std::vector<cv::Point2f> ptsPrev;
            std::vector<cv::Point2f> pts;
            // 2d vector to 1d vector
            for (unsigned int i = 0; i < fLandmarksPrev.size(); i++)
            {
               ptsPrev.insert(ptsPrev.end(), fLandmarksPrev[i].begin(), fLandmarksPrev[i].end());
            }
            std::vector<unsigned char> status;
            std::vector<float> err;
            cv::calcOpticalFlowPyrLK(grayFramePrev, grayFrame, ptsPrev, pts, status, err);
            
            // (debug) show tracked facial landmarks
            //for (unsigned int i = 0; i < pts.size(); i++)
            //{
            //   cv::circle(img, pts[i], 1, CV_RGB(255, 255, 255));
            //}
            
            // check if the tracked facial landmarks are located in a certain face
            int offset = 0;
            for (unsigned int i = 0; i < fLandmarksPrev.size(); i++)
            {
               // previous frame --> current frame
               //       i        -->    faceIdx
               int faceIdx = fp.FindLandmarksWhichFaces(pts.begin() + offset, fLandmarksPrev[i].size());
               if (faceIdx != -1)
               {
                  fp.IncFaceStatus(faceIdx, (int)faceStatusPrev[i]);
                  trackFromTo.push_back(std::pair<int, int>(i, faceIdx));
               }
               offset += fLandmarksPrev[i].size();
            }
         }
         */

        // ----------------------------
        // (debug) show faces and count
        // ----------------------------
        //if (faceNum > 0)
        //{
        //   faces = fp.GetFaces();
        //   std::vector<unsigned char> status = fp.GetFaceStatus();
        //   for (int i = 0; i < faceNum; i++)
        //   {
        //      if (status[i])
        //      {
        //         cv::rectangle(img, faces[i], CV_RGB(200, 200, 200), 2); // face with eyes
        //         cv::putText(img, std::to_string((int)status[i]), cv::Point(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2), 1, cv::FONT_HERSHEY_COMPLEX, CV_RGB(0, 0, 0), 2);
        //      }
        //      else
        //      {
        //         cv::rectangle(img, faces[i], CV_RGB(50, 50, 50), 2); // face with eyes
        //         cv::putText(img, std::to_string((int)status[i]), cv::Point(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2), 1, cv::FONT_HERSHEY_COMPLEX, CV_RGB(0, 0, 0), 2);
        //      }
        //   }
        //}

        // --------------------------------------------
        // do gender classification and display results
        // --------------------------------------------
        std::vector<unsigned char> status = fp.GetFaceStatus();
        for (int i = 0; i < faceNum; i++)
        {
            if (status[i])
            {
                cv::imshow("rrr", croppedImgs[i]);
                cv::waitKey(1);
                std::vector<Prediction> predictions = classifier.Classify(croppedImgs[i]);
                //float result = lbpSvm.Predict(croppedImgs[i]);
                // display face and gender
                //float absResult = abs(result);
                Prediction p = predictions[0];
                // show faces when the current confidence is high enough
                if (p.second >= 0.8) // 0.001 ~ 0.002
                {
                    totalCount++;
                    std::cout<<p.first<<" "<<p.second<<std::endl;
                    if (p.first == "male") // male
                    {
                        char beliefStr[64] = { 0 };
                        //sprintf(beliefStr, "%f", absResult);
                        cv::putText(img, beliefStr, cv::Point(faces[i].x, faces[i].y + faces[i].height + 30), cv::FONT_HERSHEY_COMPLEX, 0.5, CV_RGB(0, 0, 255));
                        cv::rectangle(img, faces[i], CV_RGB(0, 0, 255), 2); // male
                    }
                    else if(p.first == "female")// female
                    {
                        char beliefStr[64] = { 0 };
                        //sprintf(beliefStr, "%f", absResult);
                        cv::putText(img, beliefStr, cv::Point(faces[i].x, faces[i].y + faces[i].height + 30), cv::FONT_HERSHEY_COMPLEX, 0.5, CV_RGB(255, 0, 0));
                        cv::rectangle(img, faces[i], CV_RGB(255, 0, 0), 2); // female
                        falseCount++;
                        /*cv::imshow("FALSE", croppedImgs[i]);
                     for (int it = 0; it < (int)prevCropped.size(); it++)
                         cv::imshow(std::to_string(it), prevCropped[it]);
                     cv::waitKey();*/
                    }
                }

                // -----------------------------
                // (debug) show facial landmarks
                // -----------------------------
                /*if (showLandmark == true)
               {
                  for (int i = 0; i < faceNum; i++)
                  {
                     if (status[i])
                     {
                        for (unsigned int j = 0; j < fLandmarks[i].size(); j++)
                        {
                           cv::circle(img, fLandmarks[i][j], 1, CV_RGB(255, 255, 255), 1);
                        }
                     }
                  }
               }*/

                // --------------------------------
                // (debug) show cropped face images
                // --------------------------------
                /*if (showCroppedFaceImg == true)
               {
                  for (int i = 0; i < faceNum; i++)
                  {
                     if (status[i])
                     {
                        cv::imshow(std::to_string(i), croppedImgs[i]);
                     }
                  }
               }*/
            }
        }
        //-------------------------------------
        // For false classification's prev face
        //-------------------------------------
        /*prevCropped.clear();
           prevCropped.resize(faceNum);
           for (int i = 0; i < faceNum; i++){
               croppedImgs[i].copyTo(prevCropped[i]);
           }*/
        // ----------------------------------------------------
        // current data will be previous data in the next frame
        // ----------------------------------------------------
        /*if (faceNum > 0)
         {
            grayFrame.copyTo(grayFramePrev);
            fLandmarksPrev.resize(fLandmarks.size());
            for (unsigned int i = 0; i < fLandmarks.size(); i++)
            {
               fLandmarksPrev[i] = fLandmarks[i];
            }
            faceStatusPrev = fp.GetFaceStatus();
         }
         else
         {
            grayFramePrev = cv::Mat();
            fLandmarksPrev.clear();
            faceStatusPrev.clear();
         } */
        // show processing time
        //clock_t eTime = clock();
        tm.stop();
        double detectionTime = tm.getTimeMilli();
        double fps = 1000 / detectionTime;
        char deltaTimeStr[256] = { 0 };
        //sprintf(deltaTimeStr, "%d ms", (double)(eTime - sTime));.
        sprintf(deltaTimeStr, "%f fps", fps);
        cv::putText(img, deltaTimeStr, cv::Point(30, 30), cv::FONT_HERSHEY_COMPLEX, 1, CV_RGB(255, 255, 255));
        //cv::imshow("Result", img);
        w.showMat(img);
        //if (faceNum > 0) key = cv::waitKey();
        //else key = cv::waitKey(1);
        char key = (char)cv::waitKey(10);

        if (key == 27) break;
        else if (key == 83 || key == 115)
        {
            std::time_t time = std::time(NULL);
            char timeStr[128] = { 0 };
            std::strftime(timeStr, sizeof(timeStr), "./Offline/%Y-%m-%d-%H-%M-%S.bmp", std::localtime(&time));
            cv::imwrite(timeStr, originImg);
        }
        else if (key == 76 || key == 108) // 'l' or 'L'
        {
            showLandmark = !showLandmark;
        }
        else if (key == 70 || key == 102) // 'f' or 'F'
        {
            showCroppedFaceImg = !showCroppedFaceImg;
        }
        else if (key == 71 || key == 103)
        {
            enable_gpu = (enable_gpu ? false : true);
        }
        fp.CleanFaces();
    }
    timer.stop();
    double wholeTime = timer.getTimeSec();
    std::cout<< "time "<<wholeTime<<std::endl;
    std::cout << "False Rate :"<<falseCount <<"/"<< totalCount << std::endl;
    fly.flyCaptureStop();


    //
    return a.exec();
}
