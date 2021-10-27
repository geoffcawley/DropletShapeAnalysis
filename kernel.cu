
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include<vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

bool scaleSet = false;
bool firstframe = true;
int pixelsPerUnit = 0;
int scalePixels = 0;
int scaleUnits = 50;

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
float sqDistance(Point p1, Point p2);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

//bool process() {
//    if (!c1Defined || !c2Defined) return false;
//    twoDroplets = true;							//NOT redundant, default, see below
//    rDistance = (float)Math.sqrt((c2.cX - c1.cX) * (c2.cX - c1.cX) + (c2.cY - c1.cY) * (c2.cY - c1.cY));
//    /*DEBUG*/ System.out.println(rDistance + "jp = " + (rDistance / mData.getJpPerUnit()) + mData.getUnitType());
//    if ((rDistance * rDistance) >= ((c1.r + c2.r) * (c1.r + c2.r))) {
//        System.out.println("Two unconnected droplets");
//        InformationPanel.volumeLabel.setText(Float.toString(gv + sv));
//        tv = gv + sv;
//        return true;
//    }
//    if ((rDistance * rDistance) <= ((c1.r - c2.r) * (c1.r - c2.r))) twoDroplets = false; //above not redundant
////not sure what happens if one droplet's center lies within the other droplet
////problem for another day
//
//    /*DEBUG*/	if (twoDroplets) System.out.println("Two connected droplets");
//    /*DEBUG*/	else System.out.println("One droplet with contact bilayer");
//
//    float small = c1.r; float big = c2.r;
//    if (c2.r < c1.r) {
//        small = c2.r; big = c1.r; //}
//        /*DEBUG*/			System.out.println("Switched circles");
//    }
//    else System.out.println("No switch");
//    big /= mData.getJpPerUnit();	small /= mData.getJpPerUnit();	//needed for volume calculations
//
//    if (twoDroplets) {	//circle-circle intersection
//        //first correct distance for droplet orientation
//        float lf = rDistance / mData.getJpPerUnit();
//        // lf is apparent distance, lr is distance after compensation for 3d
//        float lr = (float)Math.sqrt((big - small) * (big - small) + lf * lf);
//        radialDistance = lr;
//        InformationPanel.distanceLabel.setText(Float.toString(lr));
//        //then do math
//                            //the following is from that droplet shape paper
//        float thetab = (float)Math.acos(
//            (lr * lr - (small * small + big * big))
//            / (2 * small * big)
//        );
//        float rdib = (small * big * (float)Math.sin(thetab)) / lr;
//        InformationPanel.dibRadius.setText(Float.toString(rdib));// / mData.getJpPerUnit()));
//        dibRadius = rdib;
//        float theta_degrees = 180.0f * thetab / (float)Math.PI; theta_degrees /= 2.0f;
//        InformationPanel.contactAngle.setText(Float.toString(theta_degrees));
//        contactAngle = theta_degrees;
//        //Dr. Lee wants half of the contact angle
//        /*DEBUG*/	System.out.println("New Method: " + rdib + ", " + Float.toString(180.0f * thetab / (float)Math.PI));
//        /*DEBUG*/	newDib = rdib;
//        //this is circle-circle intersection viewing from above
//        float lr_pixels = lr * mData.getJpPerUnit();
//        //this will cause the line denoting the circle-circle intersection to appear wrong
//        float a = ((c1.r * c1.r) - (c2.r * c2.r) + (lr_pixels * lr_pixels)) / (2.0f * lr_pixels);
//        float hi = (float)Math.sqrt(c1.r * c1.r - a * a); float b = lr_pixels - a;
//        c1h = (c1.r - a) / mData.getJpPerUnit();
//        c2h = (c2.r - b) / mData.getJpPerUnit();	//for volume truncation later
//
//        float hx = (c2.cX - c1.cX) * (a / lr_pixels) + c1.cX;
//        float hy = (c2.cY - c1.cY) * (a / lr_pixels) + c1.cY;
//        //Point P2 = P1.sub(P0).scale(a/d).add(P0);
//        i1x = hx + (hi * (c2.cY - c1.cY) / lr_pixels);
//        i1y = hy - (hi * (c2.cX - c1.cX) / lr_pixels);
//        i2x = hx - (hi * (c2.cY - c1.cY) / lr_pixels);
//        i2y = hy + (hi * (c2.cX - c1.cX) / lr_pixels);
//        iDistance = (float)Math.sqrt((i2y - i1y) * (i2y - i1y) + (i2x - i1x) * (i2x - i1x));
//        float halfpi = (float)Math.PI / 2.0f;
//        float ia1 = (float)Math.abs(Math.acos(a / c1.r));
//        float ia2 = (float)Math.abs(Math.acos(b / c2.r));
//        //source: http://stackoverflow.com/questions/3349125/circle-circle-intersection-points
//
//
//        gv = ((float)Math.PI * 4.0f * gr * gr * gr) / 3.0f;
//        gv -= ((float)Math.PI * c1h * (3.0f * rdib * rdib + c1h * c1h)) / 6.0f;
//        InformationPanel.growingVolume.setText(Float.toString(gv));
//
//        sv = ((float)Math.PI * 4.0f * sr * sr * sr) / 3.0f;
//        sv -= ((float)Math.PI * c2h * (3.0f * rdib * rdib + c2h * c2h)) / 6.0f;
//        InformationPanel.shrinkingVolume.setText(Float.toString(sv));
//
//        /*DEBUG*/ System.out.println(gr + "\t" + rdib + "\t" + c1h);
//        /*DEBUG*/ System.out.println(sr + "\t" + rdib + "\t" + c2h);
//
//        float v = ((float)Math.PI * 4.0f * big * big * big) / 3.0f;
//        v += ((float)Math.PI * 4.0f * small * small * small) / 3.0f;
//        v -= ((float)Math.PI * c1h * (3.0f * rdib * rdib + c1h * c1h)) / 6.0f;	//subtract DIB overlap
//        v -= ((float)Math.PI * c2h * (3.0f * rdib * rdib + c2h * c2h)) / 6.0f;
//        tv = v;
//        InformationPanel.volumeLabel.setText(Float.toString(v));
//    }
//    else {			//circle-surface intersection
//        if (c1.r == c2.r) return false;
//
//        //average center; contact angle requires concentric circles
//        float cX = (c1.cX + c2.cX) / 2.0f; float cY = (c1.cY + c2.cY) / 2.0f;
//        float y = (float)Math.sqrt((big * big) - (small * small));
//        float yp = -1.0f * (small / y);
//        float theta = (float)Math.abs(Math.atan(yp));
//        InformationPanel.contactAngle.setText(Float.toString(180.0f * theta / (float)Math.PI));
//
//        //dome & volume calculations
//        float h_dome = big - y;
//        float v_dome = ((float)Math.PI * h_dome * (3.0f * small * small + h_dome * h_dome)) / 6.0f;
//        float v = ((float)Math.PI * 4.0f * big * big * big) / 3.0f;
//        v -= v_dome;
//        InformationPanel.growingVolume.setText(Float.toString(v));
//        InformationPanel.shrinkingVolume.setText("N/A");
//        InformationPanel.volumeLabel.setText(Float.toString(v));
//        tv = v;
//    }
//    processed = true;
//    return true;
//}

void showImageFromFile(string fullImagePath) {
    // Read the image file
    cv::Mat image = cv::imread(fullImagePath);
    // Check for failure
    if (image.empty())
    {
        cout << "Image Not Found!!!" << endl;
        cin.get(); //wait for any key press
        return;
    }
    // Show our image inside a window.
    cv::imshow("Image Window Name here", image);

    // Wait for any keystroke in the window
    cv::waitKey(0);
}

void showVideoFromFile(string fullPath) {
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    cv::VideoCapture cap(fullPath);

    // Check if camera opened successfully
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return;
    }

    int scaleFrameStartX, scaleFrameStartY, scaleFrameEndX, scaleFrameEndY;

    int scaleLineLength = 0;
    int scaleLineStartR, scaleLineStartC, scaleLineEndC;

    Vec3i c1, c2, c1Last, c2Last;

    while (1) {

        cv::Mat frame;
        // Capture frame-by-frame
        cap >> frame;

        // Find scale line and set scale if not set already
        if (!scaleSet) {

            scaleFrameStartX = (float)frame.size().width * 0.5f;
            scaleFrameStartY = (float)frame.size().height * 0.8f;
            scaleFrameEndX = ((float)frame.size().width * 0.5f) - 1;
            scaleFrameEndY = ((float)frame.size().height * 0.2f) - 1;
            Mat scaleFrame = frame(Rect(scaleFrameStartX, scaleFrameStartY, scaleFrameEndX, scaleFrameEndY));
            cvtColor(scaleFrame, scaleFrame, COLOR_BGR2GRAY);
            threshold(scaleFrame, scaleFrame, 100, 255, THRESH_OTSU);
            bitwise_not(scaleFrame, scaleFrame);

            for (int r = 0; r < scaleFrame.rows; r++) {
                bool foundblack = false;
                int thisBlackLineLength = 0;
                int thisLineStart = 0;
                for (int c = 0; c < scaleFrame.cols; c++) {
                    if(!foundblack) {
                        if (scaleFrame.at<uchar>(r, c) == 0) {
                            thisLineStart = c;
                            thisBlackLineLength++;
                            foundblack = true; 
                        }
                    }
                    else if (foundblack) {
                        if (scaleFrame.at<uchar>(r, c) != 0) {
                            break;
                        }
                        thisBlackLineLength++;
                        if (thisBlackLineLength > scaleLineLength) {
                            scaleLineLength = thisBlackLineLength;
                            scaleLineStartR = r;
                            scaleLineStartC = thisLineStart;
                            scaleLineEndC = c;
                        }
                    }
                }
            }
            //vector<Vec4i> linesP;
            //HoughLinesP(scaleFrame, linesP, 5, CV_PI / 180, 500, 50, 10);
            //cvtColor(scaleFrame, scaleFrame, COLOR_GRAY2BGR);
            //for (int i = 0; i < 1;i++){;// linesP.size(); i++) {
            //    Vec4i l = linesP[i];
            //    line(scaleFrame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
            //}

            line(scaleFrame, Point(scaleLineStartC, scaleLineStartR), Point(scaleLineEndC, scaleLineStartR), Scalar(0, 0, 255), LINE_AA);

            //imshow("Scale Frame", scaleFrame);
            scaleSet = true;
        }

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        Mat originalFrame = frame;
        Mat dst = Mat::zeros(frame.rows, frame.cols, CV_8U);

        vector<cv::Vec3f> circles;
        //cv::Mat circles = cv::Mat();
        cv::Scalar color = cv::Scalar(255, 0, 0);
        cv::cvtColor(frame, frame, cv::COLOR_RGBA2GRAY, 0);
        double threshold = cv::threshold(frame, frame, 25, 255, THRESH_BINARY);// cv::THRESH_OTSU);
        // You can try more different parameters
        cv::HoughCircles(frame, circles, cv::HOUGH_GRADIENT, 3,
            10, 100, 50, 
            frame.rows/10, frame.rows/4);
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2RGB, 0);

        // draw 2 best circles
        // Take 2 best circles from Hough array and figure out which is which
        if (firstframe) {
            c1 = circles[0];
            c2 = circles[1];
            firstframe = false;
        }
        else {
            c1Last = c1;
            c2Last = c2;

            if (sqDistance(Point(c1[0], c1[1]), Point(circles[0][0], circles[0][1])) < sqDistance(Point(c1[0], c1[1]), Point(circles[1][0], circles[1][1]))) {
                c1 = circles[0];
                c2 = circles[1];
            }
            else {
                c1 = circles[1];
                c2 = circles[0];
            }
        }

        // draw the circle centers
        circle(frame, Point(c1[0], c1[1]), 3, Scalar(0, 255, 0), -1, 8, 0);
        circle(originalFrame, Point(c1[0], c1[1]), 3, Scalar(0, 255, 0), -1, 8, 0);
        circle(frame, Point(c2[0], c2[1]), 3, Scalar(0, 255, 0), -1, 8, 0);
        circle(originalFrame, Point(c2[0], c2[1]), 3, Scalar(0, 255, 0), -1, 8, 0);
        // draw the circle outlines
        circle(frame, Point(c1[0], c1[1]), c1[2], Scalar(0, 0, 255), 3, 8, 0);
        circle(originalFrame, Point(c1[0], c1[1]), c1[2], Scalar(0, 0, 255), 3, 8, 0);
        circle(frame, Point(c2[0], c2[1]), c2[2], Scalar(255, 0, 0), 3, 8, 0);
        circle(originalFrame, Point(c2[0], c2[1]), c2[2], Scalar(255, 0, 0), 3, 8, 0);



        // draw scale line in red
        line(frame, Point(scaleFrameStartX + scaleLineStartC, scaleFrameStartY + scaleLineStartR), Point(scaleFrameStartX + scaleLineEndC, scaleFrameStartY + scaleLineStartR), Scalar(0, 0, 255), LINE_AA);
        line(originalFrame, Point(scaleFrameStartX + scaleLineStartC, scaleFrameStartY + scaleLineStartR), Point(scaleFrameStartX + scaleLineEndC, scaleFrameStartY + scaleLineStartR), Scalar(0, 0, 255), LINE_AA);

        // Display the resulting frame
        int window_width = 800;
        int window_height = 800;
        cv::Mat resized_frame;

        //change first param to frame for the working frame, originalFrame for export
        cv::resize(frame, resized_frame, cv::Size(window_width, window_height), cv::INTER_LINEAR);
        imshow("Frame", resized_frame);

        // Press  ESC on keyboard to exit
        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    cv::destroyAllWindows();
}

int main()
{
    showVideoFromFile("C:\\Users\\geoff\\source\\repos\\DropletShapeAnalysis\\DropletShapeAnalysis\\Test Videos\\WP 30C 4POPC 1CHOL 187mosm sqe016.mp4");
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

float sqDistance(Point p1, Point p2)
{
    return (float)((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}
