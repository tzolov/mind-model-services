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

package io.mindmodel.services.twitter.sentiment;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.mindmodel.services.twitter.sentiment.domain.WordVocabulary;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tensorflow.Tensor;

import org.springframework.core.io.Resource;
import org.springframework.util.Assert;

/**
 * Converts the input Tweet JSON message into key/value map that corresponds to the Twitter Sentiment CNN model:
 * <code>
 *     data_in : vectorized TEXT tag
 *     dropout_keep_prob: 1.0f
 * </code>
 * It also preservers the original Tweet (encoded as Java Map) in the processor context. Later is used by the
 * output converter to compose the output json message.
 *
 * @author Christian Tzolov
 */
public class TwitterSentimentInputConverter implements Function<String, Map<String, Tensor>>, AutoCloseable {

	public static final Float DROPOUT_KEEP_PROB_VALUE = new Float(1.0);

	public static final String DATA_IN = "data_in";

	public static final String DROPOUT_KEEP_PROB = "dropout_keep_prob";

	private static final Log logger = LogFactory.getLog(TwitterSentimentInputConverter.class);

	private final WordVocabulary wordVocabulary;

	private final ObjectMapper objectMapper;

	public TwitterSentimentInputConverter(Resource vocabularLocation) {
		try (InputStream is = vocabularLocation.getInputStream()) {
			wordVocabulary = new WordVocabulary(is);
			objectMapper = new ObjectMapper();
			Assert.notNull(wordVocabulary, "Failed to initialize the word vocabulary");
			Assert.notNull(objectMapper, "Failed to initialize the objectMapper");
		}
		catch (IOException e) {
			throw new RuntimeException("Failed to initialize the Vocabulary", e);
		}

		logger.info("Word Vocabulary Initialized");
	}

	@Override
	public Map<String, Tensor> apply(String tweetText) {

		if (tweetText == null) {
			tweetText = "";
		}

		int[][] tweetVector = wordVocabulary.vectorizeSentence(tweetText);

		Assert.notEmpty(tweetVector, "Failed to vectorize the tweet text: " + tweetText);

		Map<String, Tensor> response = new HashMap<>();
		response.put(DATA_IN, Tensor.create(tweetVector));
		response.put(DROPOUT_KEEP_PROB, Tensor.create(DROPOUT_KEEP_PROB_VALUE));

		return response;

	}

	@Override
	public void close() throws Exception {
		logger.info("Word Vocabulary Destroyed");
		if (wordVocabulary != null) {
			wordVocabulary.close();
		}
	}
}
