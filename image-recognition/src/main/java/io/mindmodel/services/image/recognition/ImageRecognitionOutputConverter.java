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

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tensorflow.Tensor;

import org.springframework.core.io.Resource;
import org.springframework.util.Assert;
import org.springframework.util.StreamUtils;

/**
 * @author Christian Tzolov
 */
public class ImageRecognitionOutputConverter implements Function<Map<String, Tensor<Float>>, Map<String, Double>> {

	private static final Log logger = LogFactory.getLog(ImageRecognitionOutputConverter.class);

	private final List<String> labels;

	private final int responseSize;

	public ImageRecognitionOutputConverter(Resource labels, int responseSize) {
		this.responseSize = responseSize;
		try (InputStream is = labels.getInputStream()) {
			this.labels = Arrays.asList(StreamUtils.copyToString(is, Charset.forName("UTF-8")).split("\n"));
			Assert.notNull(this.labels, "Failed to initialize the labels list");
		}
		catch (IOException e) {
			throw new RuntimeException("Failed to initialize the Vocabulary", e);
		}

		logger.info("Word Vocabulary Initialized");
	}

	@Override
	public Map<String, Double> apply(Map<String, Tensor<Float>> tensorMap) {
		Tensor tensor = tensorMap.entrySet().iterator().next().getValue();
		final long[] rshape = tensor.shape();
		if (tensor.numDimensions() != 2 || rshape[0] != 1) {
			throw new RuntimeException(
					String.format("Expected model to produce a [1 N] shaped tensor where N is the number of labels, " +
							"instead it produced one with shape %s", Arrays.toString(rshape)));
		}
		int labelsCount = (int) rshape[1];

		float[][] resultMatrix = new float[1][labelsCount];

		float[] labelProbabilities = ((float[][]) tensor.copyTo(resultMatrix))[0];

		Map<String, Double> entries = new LinkedHashMap<>();
		if (responseSize == 1) {
			int maxProbabilityIndex = maxProbabilityIndex(labelProbabilities);
			System.out.println("Max:" + maxProbabilityIndex);
			entries.put(labels.get(maxProbabilityIndex), (double) labelProbabilities[maxProbabilityIndex]);
		}
		else {
			List<Integer> topKProbabilities = indexesOfTopKProbabilities(labelProbabilities, responseSize);

			for (int i = 0; i < topKProbabilities.size(); i++) {
				int probabilityIndex = topKProbabilities.get(i);
				entries.put(labels.get(probabilityIndex), (double) labelProbabilities[probabilityIndex]);
			}
		}

		return entries;
	}

	private List<Integer> indexesOfTopKProbabilities(final float[] probabilities, int k) {
		float[] copy = Arrays.copyOf(probabilities, probabilities.length);
		Arrays.sort(copy);
		float[] honey = Arrays.copyOfRange(copy, copy.length - k, copy.length);
		int[] result = new int[k];
		int resultPos = 0;
		for (int i = 0; i < probabilities.length; i++) {
			float onTrial = probabilities[i];
			int index = Arrays.binarySearch(honey, onTrial);
			if (index < 0) continue;
			result[resultPos++] = i;
		}
		return sortByProb(probabilities, result);
	}

	private List<Integer> sortByProb(final float[] probabilities, int[] topK) {
		List<Integer> topKList = Arrays.stream(topK).boxed().collect(Collectors.toList());
		Collections.sort(topKList, (o1, o2) -> {
			if (probabilities[o1] == probabilities[o2]) {
				return 0;
			}
			else if (probabilities[o1] < probabilities[o2]) {
				return 1;
			}
			return -1;
		});

		return topKList;
	}

	private int maxProbabilityIndex(float[] probabilities) {
		int best = 0;
		for (int i = 1; i < probabilities.length; ++i) {
			if (probabilities[i] > probabilities[best]) {
				best = i;
			}
		}
		return best;
	}

}
