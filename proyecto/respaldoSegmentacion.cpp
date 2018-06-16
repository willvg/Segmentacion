//Para la segmentacion de partes
        
       cv::Mat src = image;
       
       for( int x = 0; x < src.rows; x++ ) {
          for( int y = 0; y < src.cols; y++ ) {
              if ( src.at<cv::Vec3b>(x, y) == cv::Vec3b(255,255,255) ) {
                src.at<cv::Vec3b>(x, y)[0] = 0;
                src.at<cv::Vec3b>(x, y)[1] = 0;
                src.at<cv::Vec3b>(x, y)[2] = 0;
              }
            }
        }
        // Show output image
        imshow("Black Background Image", src);
        // Create a kernel that we will use for accuting/sharpening our image
        cv::Mat kernel = (cv::Mat_<float>(3,3) <<
                1,  1, 1,
                1, -8, 1,
                1,  1, 1); // an approximation of second derivative, a quite strong kernel
        // do the laplacian filtering as it is
        // well, we need to convert everything in something more deeper then CV_8U
        // because the kernel has some negative values,
        // and we can expect in general to have a Laplacian image with negative values
        // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
        // so the possible negative number will be truncated
        cv::Mat imgLaplacian;
        cv::Mat sharp = src; // copy source image to another temporary one
        cv::filter2D(sharp, imgLaplacian, CV_32F, kernel);
        src.convertTo(sharp, CV_32F);
        cv::Mat imgResult = sharp - imgLaplacian;
        // convert back to 8bits gray scale
        imgResult.convertTo(imgResult, CV_8UC3);
        imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
        // imshow( "Laplace Filtered Image", imgLaplacian );
        //cv::imshow( "New Sharped Image", imgResult );
        src = imgResult; // copy back
        // Create binary image from source image
        cv::Mat bw;
        cv::cvtColor(src, bw, CV_BGR2GRAY);
        cv::threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        cv::imshow("Binary Image", bw);

        // Perform the distance transform algorithm
        cv::Mat dist;
        cv::distanceTransform(bw, dist, CV_DIST_L2, 3);
        // Normalize the distance image for range = {0.0, 1.0}
        // so we can visualize and threshold it
        cv::normalize(dist, dist, 0, 1., cv::NORM_MINMAX);
        cv::imshow("Distance Transform Image", dist);
        // Threshold to obtain the peaks
        // This will be the markers for the foreground objects
        cv::threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
        // Dilate a bit the dist image
        cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8UC1);
        cv::dilate(dist, dist, kernel1);
        cv::imshow("Peaks", dist);
        // Create the CV_8U version of the distance image
        // It is needed for findContours()
        cv::Mat dist_8u;
        dist.convertTo(dist_8u, CV_8U);
        // Find total markers
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        // Create the marker image for the watershed algorithm
        cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32SC1);
        // Draw the foreground markers
        for (size_t i = 0; i < contours.size(); i++)
            cv::drawContours(markers, contours, static_cast<int>(i), cv::Scalar::all(static_cast<int>(i)+1), -1);
        // Draw the background marker
        cv::circle(markers, cv::Point(5,5), 3, CV_RGB(255,255,255), -1);
        cv::imshow("Markers", markers*10000);
        // Perform the watershed algorithm
        cv::watershed(src, markers);
        cv::Mat mark = cv::Mat::zeros(markers.size(), CV_8UC1);
        markers.convertTo(mark, CV_8UC1);
        cv::bitwise_not(mark, mark);
    //    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
                                      // image looks like at that point
        // Generate random colors
        std::vector<cv::Vec3b> colors;
        for (size_t i = 0; i < contours.size(); i++)
        {
            int b = cv::theRNG().uniform(0, 255);
            int g = cv::theRNG().uniform(0, 255);
            int r = cv::theRNG().uniform(0, 255);
            colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
        }
        // Create the result image
        cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);

        // Fill labeled objects with random colors
        for (int i = 0; i < markers.rows; i++)
        {
            for (int j = 0; j < markers.cols; j++)
            {
                int index = markers.at<int>(i,j);
                if (index > 0 && index <= static_cast<int>(contours.size()))
                    dst.at<cv::Vec3b>(i,j) = colors[index-1];
                else
                    dst.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
            }
        }
        // Visualize the final image
    imshow("Final Result", dst);
        