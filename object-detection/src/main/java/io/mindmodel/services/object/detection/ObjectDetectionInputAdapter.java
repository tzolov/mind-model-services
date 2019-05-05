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

package io.mindmodel.services.object.detection;

import java.util.Collections;
import java.util.Map;

import io.mindmodel.services.common.GraphInputAdapter;
import io.mindmodel.services.common.GraphRunner;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.op.io.TfRecordReader;
import org.tensorflow.types.UInt8;

/**
 * Converts byte array image into a input Tensor for the Object Detection API.
 *
 * @author Christian Tzolov
 */
public class ObjectDetectionInputAdapter implements GraphInputAdapter<byte[]> {

	private static final Log logger = LogFactory.getLog(ObjectDetectionInputAdapter.class);

	public static final String RAW_IMAGE = "raw_image";
	public static final String NORMALIZED_IMAGE = "normalized_image";
	private static final long CHANNELS = 3;

	private final GraphRunner imageLoaderGraph;

	public ObjectDetectionInputAdapter() {

		this.imageLoaderGraph = new GraphRunner(RAW_IMAGE, NORMALIZED_IMAGE) {
			@Override
			protected void doGraphDefinition(Ops tf) {
				Placeholder<String> rawImage = tf.withName(RAW_IMAGE).placeholder(String.class);
				Operand<UInt8> decodedImage = tf.dtypes.cast(
						tf.image.decodeJpeg(rawImage, DecodeJpeg.channels(CHANNELS)), UInt8.class);
				// Expand dimensions since the model expects images to have shape: [1, H, W, 3]
				tf.withName(NORMALIZED_IMAGE).expandDims(decodedImage, tf.constant(0));
			}
		};
	}

	@Override
	public Map<String, Tensor<?>> apply(byte[] inputImage) {
		try (Tensor inputTensor = Tensor.create(inputImage)) {
			return this.imageLoaderGraph.apply(Collections.singletonMap(RAW_IMAGE, inputTensor));
		}
	}

	@Override
	public void close() {
		if (this.imageLoaderGraph != null) {
			this.imageLoaderGraph.close();
		}
	}
}
