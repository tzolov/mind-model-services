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

package io.mindmodel.services.pose.estimation;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;
import java.util.function.Function;

import javax.imageio.ImageIO;

import io.mindmodel.services.common.GraphicsUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tensorflow.Tensor;


/**
 * Converts byte array image into a input Tensor for the Pose Estimation API. The computed image tensors uses the
 * 'image' model placeholder.
 *
 * @author Christian Tzolov
 */
public class PoseEstimationTensorflowInputConverter implements Function<byte[][], Map<String, Tensor<?>>> {

	private static final Log logger = LogFactory.getLog(PoseEstimationTensorflowInputConverter.class);

	private static final long BATCH_SIZE = 1;
	private static final long CHANNELS = 3;
	public static final String IMAGE_TENSOR_FEED_NAME = "image";
	private final static int[] COLOR_CHANNELS = new int[] { 0, 1, 2 };

	private boolean debugVisualizationEnabled = false;

	public boolean isDebugVisualizationEnabled() {
		return debugVisualizationEnabled;
	}

	public void setDebugVisualizationEnabled(boolean debugVisualizationEnabled) {
		this.debugVisualizationEnabled = debugVisualizationEnabled;
	}

	@Override
	public Map<String, Tensor<?>> apply(byte[][] imageBytesArray) {
		try {
			int batchSize = imageBytesArray.length;
			FloatBuffer floatBuffer = null;
			long[] shape = null;
			for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
				byte[] imageBytes = imageBytesArray[batchIndex];

				ByteArrayInputStream is = new ByteArrayInputStream(imageBytes);
				BufferedImage img = ImageIO.read(is);

				if (img.getType() != BufferedImage.TYPE_INT_ARGB) {
					img = GraphicsUtils.toBufferedImageType(img, BufferedImage.TYPE_INT_ARGB);
				}

				if (floatBuffer == null) {
					floatBuffer = FloatBuffer.allocate((int) (batchSize * img.getHeight() * img.getWidth() * CHANNELS));
					shape = new long[] { batchSize, img.getHeight(), img.getWidth(), CHANNELS };
				}

				// ImageIO.read produces BGR-encoded images, while the model expects RGB.
				int[] data = ((DataBufferInt) img.getRaster().getDataBuffer()).getData();
				float[] dataFloat = toRgbFloat(data);
				floatBuffer.put(dataFloat);
			}
			floatBuffer.flip();

			//if (this.debugVisualizationEnabled) {
			//	processorContext.put("inputImage", input);
			//}

			Tensor tensor = Tensor.create(shape, floatBuffer);
			return Collections.singletonMap(IMAGE_TENSOR_FEED_NAME, tensor);
		}
		catch (IOException e) {
			throw new IllegalArgumentException("Incorrect image format", e);
		}
	}

	private float[] toRgbFloat(int[] data) {
		float[] float_image = new float[data.length * 3];
		for (int i = 0; i < data.length; ++i) {
			final int val = data[i];
			float_image[i * 3 + COLOR_CHANNELS[0]] = ((val >> 16) & 0xFF); //R
			float_image[i * 3 + COLOR_CHANNELS[1]] = ((val >> 8) & 0xFF);  //G
			float_image[i * 3 + COLOR_CHANNELS[2]] = (val & 0xFF);         //B
		}
		return float_image;
	}
}
