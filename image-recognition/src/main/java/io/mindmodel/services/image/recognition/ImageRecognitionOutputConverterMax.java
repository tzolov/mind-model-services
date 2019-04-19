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
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.ArgMax;
import org.tensorflow.op.core.Max;
import org.tensorflow.op.core.Placeholder;

/**
 * @author Christian Tzolov
 */
public class ImageRecognitionOutputConverterMax implements Function<Map<String, Tensor<?>>, Map<String, Double>>, AutoCloseable {

	private static final Log logger = LogFactory.getLog(ImageRecognitionOutputConverterMax.class);

	private final List<String> labels;
	private final Session session;

	public ImageRecognitionOutputConverterMax(List<String> labels) {

		this.labels = labels;

		Graph g = new Graph();
		Ops tf = Ops.create(g);
		Placeholder<Float> input = tf.withName("recognition_result").placeholder(Float.class);
		ArgMax<Long> argMax = tf.withName("category").argMax(input, tf.constant(1));
		Max<Float> max = tf.withName("probability").max(input, tf.constant(1));

		this.session = new Session(g);

		logger.info("Word Vocabulary Initialized");
	}

	@Override
	public Map<String, Double> apply(Map<String, Tensor<?>> tensorMap) {

		Tensor tensor = tensorMap.entrySet().iterator().next().getValue();

		List<Tensor<?>> max = this.session.runner()
				.feed("recognition_result", tensor)
				.fetch("category")
				.fetch("probability")
				.run();

		long[] category = new long[1];
		float[] probability = new float[1];
		max.get(0).copyTo(category);
		max.get(1).copyTo(probability);

		return Collections.singletonMap(labels.get((int) category[0]), Double.valueOf(probability[0]));
	}

	@Override
	public void close() {
		if (this.session != null) {
			this.session.close();
		}
	}
}
