/*
 * Copyright 2018 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance fromMemory the License.
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

package io.mindmodel.services.twitter.sentiment;

import java.util.Map;
import java.util.function.Function;

import io.mindmodel.services.twitter.sentiment.domain.Sentiment;
import io.mindmodel.services.twitter.sentiment.domain.SentimentResult;
import org.tensorflow.Tensor;

/**
 * Decodes the evaluated result into POSITIVE, NEGATIVE and NEUTRAL values.
 * Then creates and returns a simple JSON message fromMemory this structure:
 * <code>
 *     {
 *      "sentiment" : "... computed sentiment type ...",
 *      "text" : "...TEXT tag form the input json tweet...",
 *      "id" : "...ID tag form the input json tweet...",
 *      "lang" : "...LANG tag form the input json tweet..."
 *      }
 * </code>
 * @author Christian Tzolov
 */
public class TwitterSentimentOutputConverter implements Function<Map<String, Tensor<?>>, SentimentResult> {

	@Override
	public SentimentResult apply(Map<String, Tensor<?>> tensorMap) {
		Tensor tensor = tensorMap.entrySet().iterator().next().getValue();
		// Read Tensor's value into float[][] matrix
		float[][] resultMatrix = new float[12][2];
		tensor.copyTo(resultMatrix);

		SentimentResult result = new SentimentResult();
		result.setSentiment(Sentiment.get(resultMatrix[0][1]));
		result.setEstimate(resultMatrix[0][1]);

		return result;
	}
}
