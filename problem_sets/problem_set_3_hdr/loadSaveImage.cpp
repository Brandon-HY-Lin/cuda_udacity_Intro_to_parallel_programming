void loadImageHDR (const std::string &filename,
					float **imagePtr,
					std::size_t *numRows, std::size_t *numCols)
{
	cv::Mat originImg = cv::imread(filename.c_str(), cv::IMREAD_COLOR);

	cv::Mat image;
	
	if (originImg.type() != CV_32FC3) {
		originImg.convertTo(image, CV_32FC3);
	} else {
		image = originImg;
	}
	
	if (image.empty()) {
	    std::cerr << "Couldn't open file: " << filename << std::endl;
	    exit(1);
	}

  if (image.channels() != 3) {
    std::cerr << "Image must be color!" << std::endl;
    exit(1);
  }

  if (!image.isContinuous()) {
    std::cerr << "Image isn't continuous!" << std::endl;
    exit(1);
  }
	*imagePtr = new float[image.rows * image.cols * image.channels()];
	
	// copy image
	float *cvPtr = image.ptr<float>(0);
	for (std::size_t i = 0; i < image.rows * image.cols * image.channels(); ++i)
		(*imagePtr)[i] = cvPtr[i];
  *numRows = image.rows;
  *numCols = image.cols;
}