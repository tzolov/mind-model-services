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

import java.util.Collections;
import java.util.Map;
import java.util.function.Function;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.DecodeJpeg;
import org.tensorflow.op.core.Placeholder;


/**
 * @author Christian Tzolov
 */
public class ImageRecognitionInputConverter implements Function<byte[], Map<String, Tensor>>, AutoCloseable {

	private static final Log logger = LogFactory.getLog(ImageRecognitionInputConverter.class);
	public static final String NORMALIZE_IMAGE_GRAPH_INPUT_NAME = "raw_image_input";
	public static final String NORMALIZE_IMAGE_GRAPH_OUTPUT_NAME = "normalized_image";

	//private final Output graphOutput;
	private final String inceptionModelInputNodeName;
	private final Session session;

	/**
	 * Normalizes the raw input image into format expected by the pre-trained Inception/MobileNetV1/MobileNetV2 models.
	 * Typically the model is trained with images scaled to certain size. Usually it is 224x224 pixels, but can be
	 * also 192x192, 160x160, 128128, 92x92. Use the (imageHeight, imageWidth) to set the desired size.
	 * The colors, represented as R, G, B in 1-byte each were converted to float using (Value - Mean)/Scale.
	 *
	 * @param inceptionModelInputNodeName name of the input node in the pre-trained model.
	 * @param imageHeight normalized image height.
	 * @param imageWidth normalized image width.
	 * @param mean mean value to normalize the input image.
	 * @param scale scale to normalize the input image.
	 */
	public ImageRecognitionInputConverter(String inceptionModelInputNodeName,
			int imageHeight, int imageWidth, float mean, float scale) {
		this.inceptionModelInputNodeName = inceptionModelInputNodeName;
		this.session = this.buildNormalizeImageGraph(imageHeight, imageWidth, mean, scale);
	}

	@Override
	public Map<String, Tensor> apply(byte[] input) {
		return Collections.singletonMap(this.inceptionModelInputNodeName, this.normalizeImage(input));
	}

	private Tensor normalizeImage(byte[] inputImage) {
		try (Tensor inputTensor = Tensor.create(inputImage)) {
			return this.session.runner()
					.feed(NORMALIZE_IMAGE_GRAPH_INPUT_NAME, inputTensor)
					.fetch(NORMALIZE_IMAGE_GRAPH_OUTPUT_NAME)
					.run().get(0);
		}
	}

	private Session buildNormalizeImageGraph(int imageHeight, int imageWidth, float mean, float scale) {

		Graph g = new Graph();

		Ops tf = Ops.create(g);  // Normalize image eagerly, using default session

		Placeholder<String> input = tf.withName(NORMALIZE_IMAGE_GRAPH_INPUT_NAME).placeholder(String.class);

		final Operand<Float> decodedImage =
				tf.cast(tf.decodeJpeg(input.asOutput(), DecodeJpeg.channels(3L)), Float.class);

		final Operand<Float> resizedImage =
				tf.resizeBilinear(
						tf.expandDims(decodedImage, tf.constant(0)),
						tf.constant(new int[] { imageHeight, imageWidth }));

		Operand<Float> normalizeOperand = tf.withName(NORMALIZE_IMAGE_GRAPH_OUTPUT_NAME)
				.div(tf.sub(resizedImage, tf.constant(mean)), tf.constant(scale));

		return new Session(g);
	}

	@Override
	public void close() {
		logger.info("Input Graph Destroyed");
		if (this.session != null) {
			session.close();
		}
	}
}
