/*
 * Copyright 2017-2018 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.mindmodel.services.image.recognition;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import io.mindmodel.services.common.GraphRunner;
import io.mindmodel.services.common.GraphInputAdapter;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.image.DecodeJpeg;


/**
 * @author Christian Tzolov
 */
public class ImageRecognitionInputAdapter implements GraphInputAdapter<byte[]>, AutoCloseable {

	private static final Log logger = LogFactory.getLog(ImageRecognitionInputAdapter.class);

	public static final String RAW_IMAGE = "raw_image";
	public static final String NORMALIZED_IMAGE = "normalized_image";

	private final GraphRunner imageNormalizationGraphRunner;

	/**
	 * Normalizes the raw input image into format expected by the pre-trained Inception/MobileNetV1/MobileNetV2 models.
	 * Typically the model is trained with images scaled to certain size. Usually it is 224x224 pixels, but can be
	 * also 192x192, 160x160, 128128, 92x92. Use the (imageHeight, imageWidth) to set the desired size.
	 * The colors, represented as R, G, B in 1-byte each were converted to float using (Value - Mean)/Scale.
	 *
	 * @param imageHeight normalized image height.
	 * @param imageWidth normalized image width.
	 * @param mean mean value to normalize the input image.
	 * @param scale scale to normalize the input image.
	 */
	public ImageRecognitionInputAdapter(final int imageHeight, final int imageWidth, final float mean, final float scale) {

		this.imageNormalizationGraphRunner = new GraphRunner(
				Arrays.asList(RAW_IMAGE),
				Arrays.asList(NORMALIZED_IMAGE)) {

			@Override
			protected void doGraphDefinition(Ops tf) {
				Placeholder<String> input = tf.withName(RAW_IMAGE).placeholder(String.class);

				final Operand<Float> decodedImage =
						tf.dtypes.cast(tf.image.decodeJpeg(input.asOutput(), DecodeJpeg.channels(3L)), Float.class);

				final Operand<Float> resizedImage = tf.image.resizeBilinear(
						tf.expandDims(decodedImage, tf.constant(0)),
						tf.constant(new int[] { imageHeight, imageWidth }));

				Operand<Float> normalizedImage = tf.withName(NORMALIZED_IMAGE)
						.math.div(tf.math.sub(resizedImage, tf.constant(mean)), tf.constant(scale));
			}
		};
	}

	@Override
	public Map<String, Tensor<?>> apply(byte[] inputImage) {
		try (Tensor inputTensor = Tensor.create(inputImage)) {
			return this.imageNormalizationGraphRunner.apply(Collections.singletonMap(RAW_IMAGE, inputTensor));
		}
	}

	@Override
	public void close() {
		logger.info("Input Graph Destroyed");
		if (this.imageNormalizationGraphRunner != null) {
			imageNormalizationGraphRunner.close();
		}
	}
}
