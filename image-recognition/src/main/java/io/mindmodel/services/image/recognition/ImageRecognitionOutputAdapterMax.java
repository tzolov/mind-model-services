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
import java.util.List;
import java.util.Map;

import io.mindmodel.services.common.AutoCloseables;
import io.mindmodel.services.common.GraphRunner;
import io.mindmodel.services.common.GraphOutputAdapter;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Max;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.math.ArgMax;

/**
 * @author Christian Tzolov
 */
public class ImageRecognitionOutputAdapterMax implements GraphOutputAdapter<Map<String, Double>>, AutoCloseable {

	private static final Log logger = LogFactory.getLog(ImageRecognitionOutputAdapterMax.class);

	private final List<String> labels;
	private final GraphRunner graphRunner;

	public ImageRecognitionOutputAdapterMax(List<String> labels) {

		this.labels = labels;

		this.graphRunner = new GraphRunner(Arrays.asList("recognition_result"),
				Arrays.asList("category", "probability")) {
			@Override
			protected void doGraphDefinition(Ops tf) {
				Placeholder<Float> input = tf.withName("recognition_result").placeholder(Float.class);
				ArgMax<Long> argMax = tf.withName("category").math.argMax(input, tf.constant(1));
				Max<Float> max = tf.withName("probability").max(input, tf.constant(1));
			}
		};

		logger.info("Word Vocabulary Initialized");
	}

	@Override
	public Map<String, Double> apply(Map<String, Tensor<?>> tensorMap) {

		Tensor tensor = tensorMap.entrySet().iterator().next().getValue();

		Map<String, Tensor<?>> max = this.graphRunner.apply(Collections.singletonMap("recognition_result", tensor));

		long[] category = new long[1];
		float[] probability = new float[1];
		max.get("category").copyTo(category);
		max.get("probability").copyTo(probability);

		AutoCloseables.all(max);

		return Collections.singletonMap(labels.get((int) category[0]), Double.valueOf(probability[0]));
	}

	@Override
	public void close() {
		if (this.graphRunner != null) {
			this.graphRunner.close();
		}
	}
}
