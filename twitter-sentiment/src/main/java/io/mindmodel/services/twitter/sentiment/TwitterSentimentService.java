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

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import io.mindmodel.services.common.attic.TensorFlowService;
import io.mindmodel.services.twitter.sentiment.domain.SentimentResult;

import org.springframework.core.io.DefaultResourceLoader;

/**
 * @author Christian Tzolov
 */
public class TwitterSentimentService {

	public static final List<String> FETCH_NAMES = Arrays.asList("output/Softmax");
	public static final String DEFAULT_MODEL_URI = "http://dl.bintray.com/big-data/generic/minimal_graph.proto";
	public static final String DEFAULT_VOCABULARY_URI = "http://dl.bintray.com/big-data/generic/vocab.csv";

	private final Function<Object, SentimentResult> sentimentJsonDetector;
	private final Function<String, SentimentResult> sentimentTextDetector;

	private final TwitterSentimentInputConverter inputConverter;
	private final TwitterSentimentOutputConverter outputConverter;
	private final TweetTagExtractor tweetTextTagExtractor;
	private final TensorFlowService tensorFlowService;

	public TwitterSentimentService() {
		this(DEFAULT_MODEL_URI, DEFAULT_VOCABULARY_URI, true);
	}

	public TwitterSentimentService(String modelUri, String vocabularyUri, boolean cacheModel) {
		this(new TwitterSentimentInputConverter(new DefaultResourceLoader().getResource(vocabularyUri)),
				new TwitterSentimentOutputConverter(),
				new TweetTagExtractor(TweetTagExtractor.TWEET_TEXT_TAG),
				new TensorFlowService(new DefaultResourceLoader().getResource(modelUri), FETCH_NAMES, cacheModel));
	}

	public TwitterSentimentService(TwitterSentimentInputConverter inputConverter,
			TwitterSentimentOutputConverter outputConverter,
			TweetTagExtractor tweetTagExtractor,
			TensorFlowService tensorFlowService) {
		this.inputConverter = inputConverter;
		this.outputConverter = outputConverter;
		this.tweetTextTagExtractor = tweetTagExtractor;
		this.tensorFlowService = tensorFlowService;

		this.sentimentTextDetector = this.inputConverter.andThen(this.tensorFlowService).andThen(this.outputConverter);
		this.sentimentJsonDetector = this.tweetTextTagExtractor.andThen(sentimentTextDetector);
	}

	public SentimentResult tweetSentiment(Object jsonTweet) {
		return sentimentJsonDetector.apply(jsonTweet);
	}

	public SentimentResult textSentiment(String text) {
		return sentimentTextDetector.apply(text);
	}
}
