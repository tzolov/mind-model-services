/*
 * Copyright 2018 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.mindmodel.services.semantic.segmentation;

import java.awt.image.BufferedImage;
import java.util.Collections;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

import io.mindmodel.services.common.GraphicsUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

/**
 *
 * @author Christian Tzolov
 */
public class SemanticSegmentationConfiguration {

	private static final Log logger = LogFactory.getLog(SemanticSegmentationConfiguration.class);

	private static final int BATCH_SIZE = 1;

	/**
	 * Blended mask transparency. Value is between 0.0 (0% transparency) and 1.0 (100% transparent).
	 */
	private double maskTransparency = 0.3;

	/**
	 * Generated image format
	 */
	private String imageFormat = "jpg";

	public double getMaskTransparency() {
		return maskTransparency;
	}

	public void setMaskTransparency(double maskTransparency) {
		this.maskTransparency = maskTransparency;
	}

	public String getImageFormat() {
		return imageFormat;
	}

	public void setImageFormat(String imageFormat) {
		this.imageFormat = imageFormat;
	}

	/**
	 * Converts the input image (as byte[]) into input tensor
	 * @return
	 */
	public Function<byte[], Map<String, Tensor<?>>> inputConverter() {
		return image -> {
			BufferedImage scaledImage = SemanticSegmentationUtils.scaledImage(image);
			Tensor<UInt8> inTensor = SemanticSegmentationUtils.createInputTensor(scaledImage);
			return Collections.singletonMap(SemanticSegmentationUtils.INPUT_TENSOR_NAME, inTensor);
		};
	}

	/**
	 * Converts output named tensors into pixel masks
	 * @return
	 */
	public Function<Map<String, Tensor<?>>, long[][]> outputConverter() {
		return resultTensors -> {
			Tensor<?> outputTensor = resultTensors.get(SemanticSegmentationUtils.OUTPUT_TENSOR_NAME);
			int width = (int) outputTensor.shape()[1];
			int height = (int) outputTensor.shape()[2];
			long[][] maskPixels = outputTensor.copyTo(new long[BATCH_SIZE][width][height])[0];
			return maskPixels;
		};
	}

	/**
	 * Takes the input image (byte[]) and mask pixels (long[][]) and outputs the same image (byte[]) augmented
	 * with masks overlays.
	 * @return Returns the input image augmented with masks's overlays
	 */
	public BiFunction<byte[], long[][], byte[]> imageAugmenter() {
		return (inputImage, mask) -> {
			int[][] maskPixels = SemanticSegmentationUtils.toIntArray(mask);

			try {
				int height = maskPixels.length;
				int width = maskPixels[0].length;

				BufferedImage scaledImage = SemanticSegmentationUtils.scaledImage(inputImage);

				BufferedImage maskImage = SemanticSegmentationUtils.createMaskImage(
						maskPixels, width, height, this.getMaskTransparency());

				BufferedImage blend = SemanticSegmentationUtils.blendMask(maskImage, scaledImage);

				return GraphicsUtils.toImageByteArray(blend, this.getImageFormat());
			}
			catch (Exception e) {
				logger.error("Failed to create output message", e);
			}
			return inputImage;
		};
	}

	/**
	 * Converts the pixels (long[][]) into mask image (byte[])
	 * @return Image representing the mask pixels
	 */
	public Function<long[][], byte[]> pixelsToMaskImage() {
		return maskPixels -> {
			try {

				int height = maskPixels.length;
				int width = maskPixels[0].length;

				BufferedImage maskImage = SemanticSegmentationUtils.createMaskImage(
						SemanticSegmentationUtils.toIntArray(maskPixels), width, height, this.getMaskTransparency());

				return GraphicsUtils.toImageByteArray(maskImage, this.getImageFormat());
			}
			catch (Exception e) {
				logger.error("Failed to create output message", e);
			}
			return new byte[0];
		};
	}

}
