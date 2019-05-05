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
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import io.mindmodel.services.common.AutoCloseables;
import io.mindmodel.services.common.GraphRunner;
import io.mindmodel.services.common.GraphOutputAdapter;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.nn.TopK;

/**
 * @author Christian Tzolov
 */
public class ImageRecognitionOutputAdapterTopK implements GraphOutputAdapter<Map<String, Double>>, AutoCloseable {

	private static final Log logger = LogFactory.getLog(ImageRecognitionOutputAdapterTopK.class);

	private final List<String> labels;

	private final GraphRunner graphRunner;

	public ImageRecognitionOutputAdapterTopK(List<String> labels, int responseSize) {
		this.labels = labels;
		this.graphRunner = new GraphRunner(Arrays.asList("recognition_result"),
				Arrays.asList("topK")) {
			@Override
			protected void doGraphDefinition(Ops tf) {
				Placeholder<Float> input = tf.withName("recognition_result").placeholder(Float.class);
				TopK<Float> topK = tf.withName("topK").nn.topK(input, tf.constant(responseSize), TopK.sorted(true));
			}
		};
	}

	@Override
	public Map<String, Double> apply(Map<String, Tensor<?>> tensorMap) {

		Tensor responseTensor = tensorMap.entrySet().iterator().next().getValue();
		float[][] results = new float[(int) responseTensor.shape()[0]][(int) responseTensor.shape()[1]];
		responseTensor.copyTo(results);

		Map<String, Tensor<?>> detectedImages = this.graphRunner.apply(
				Collections.singletonMap("recognition_result", responseTensor));

		Tensor<Float> topKTensor = detectedImages.get("topK").expect(Float.class);
		float[][] topK = new float[(int) topKTensor.shape()[0]][(int) topKTensor.shape()[1]];
		topKTensor.copyTo(topK);

		float min = topK[0][topK[0].length - 1];

		Map<Float, Integer> valueToIndex = new HashMap<>();
		for (int i = 0; i < results[0].length; i++) {
			if (results[0][i] >= min)
				valueToIndex.put(results[0][i], i);
		}

		Map<String, Double> map = new LinkedHashMap<>();
		for (float tk : topK[0]) {
			map.put(labels.get(valueToIndex.get(tk)), (double) tk);
		}

		AutoCloseables.all(detectedImages);

		return map;
	}

	@Override
	public void close() {
		if (this.graphRunner != null) {
			this.graphRunner.close();
		}
	}
}
