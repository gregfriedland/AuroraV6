#include <iostream>
#include "Util.h"
#include "Camera.h"
#include <fstream>
#include <unistd.h>
#include <thread>

Camera::Camera(const CameraSettings& settings)
: m_fpsCounter(30000, "Camera"), m_settings(settings) {
    m_cam.set(CV_CAP_PROP_FORMAT, CV_8UC3);

    m_cam.set(CV_CAP_PROP_FRAME_WIDTH, m_settings.m_camWidth);
    m_cam.set(CV_CAP_PROP_FRAME_HEIGHT, m_settings.m_camHeight);
    m_cam.set(CV_CAP_PROP_FPS, m_settings.m_fps);
}

int Camera::camWidth() const { 
    return m_settings.m_camWidth;
}

int Camera::camHeight() const { 
    return m_settings.m_camHeight;
}

void Camera::setImageProcSettings(const ImageProcSettings& settings) {
    std::cout << "Setting image proc settings: " << settings.toString() << std::endl;
    m_imageProcSettings = settings;
}

void Camera::init() {
#ifdef RASPICAM
    if (!m_cam.open()) {
#else
    if (!m_cam.open(0)) {
#endif
        std::cerr << "Error opening camera" << std::endl;
        return;
    }
}

void Camera::start(unsigned int interval) {
    init();

    std::cout << "Starting camera with dims " << m_settings.m_camWidth << "x" << m_settings.m_camHeight << "\n";
    m_stop = false;
    auto run = [=]() {
        while (!m_stop) {
            loop(interval);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        m_stop = false;
    };
    m_thread = std::thread(run);
    m_thread.detach();
}

void Camera::stop() {
	std::cout << "Stopping camera\n";
    m_stop = true;
    if (m_thread.joinable()) {
        m_thread.join();
    }
}	

cv::Mat Camera::getGrayImage() {
    m_mutex.lock();
    auto img = m_grayImg.clone();
    m_mutex.unlock();
    return img;
}

cv::Mat Camera::getScaledImage() {
    m_mutex.lock();
    auto img = m_screenImg.clone();
    m_mutex.unlock();
    return img;
}

void Camera::loop(unsigned int interval) {
    m_frameTimer.tick(interval, [=]() {
        m_fpsCounter.tick();

            m_cam.grab();
            m_cam.retrieve(m_img); // get a new frame from camera

	    m_mutex.lock();
            cv::cvtColor(m_img, m_grayImg, CV_BGR2GRAY);
	    m_mutex.unlock();

	    cv::Mat screenImg;
            cv::resize(m_grayImg, screenImg,
		       cv::Size(m_imageProcSettings.m_intermediateResizeFactor * m_settings.m_screenWidth,
				m_imageProcSettings.m_intermediateResizeFactor * m_settings.m_screenHeight));
            cv::medianBlur(screenImg, screenImg, 2 * m_imageProcSettings.m_medianBlurSize + 1); // blur without losing edges
            // cv::GaussianBlur(m_screenImg, m_screenImg, cv::Size(3, 3), 0, 0); // blur

            if (m_imageProcSettings.m_morphOperation >= 0) {
                if (m_imageProcSettings.m_morphOperation > 4) {
                    std::cout << "Invalid morph operation\n";
                } else {
                    cv::Mat element = cv::getStructuringElement(m_imageProcSettings.m_morphKernel,
                    cv::Size(2 * m_imageProcSettings.m_morphKernelSize + 1, 2 * m_imageProcSettings.m_morphKernelSize + 1),
                    cv::Size(m_imageProcSettings.m_morphKernelSize, m_imageProcSettings.m_morphKernelSize));
                    cv::morphologyEx(screenImg, screenImg, m_imageProcSettings.m_morphOperation + 2, element);
                }
            }
        screenImg.convertTo(screenImg, -1, m_imageProcSettings.m_contrastFactor, 0); // increase contrast
        cv::medianBlur(screenImg, screenImg, 2 * m_imageProcSettings.m_medianBlurSize + 1); // blur without losing edges
	    cv::resize(screenImg, screenImg, cv::Size(m_settings.m_screenWidth, m_settings.m_screenHeight));

	    m_mutex.lock();
	    m_screenImg = screenImg;
	    //	    cv::resize(m_grayImg, m_screenImg, cv::Size(m_settings.m_screenWidth, m_settings.m_screenHeight));
	    m_mutex.unlock();
    });
}

 void Camera::lock() {
   m_mutex.lock();
 }

 void Camera::unlock() {
   m_mutex.unlock();
 }
