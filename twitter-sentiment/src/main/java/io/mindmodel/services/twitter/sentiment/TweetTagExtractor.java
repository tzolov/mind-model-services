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
import java.util.Map;
import java.util.function.Function;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.springframework.util.Assert;

import static org.springframework.util.StringUtils.isEmpty;

/**
 * Extracts the text tag from the input Tweet JSON message
 *
 * @author Christian Tzolov
 */
public class TweetTagExtractor implements Function<Object, String> {

	public static final String TWEET_TEXT_TAG = "text";

	public static final String TWEET_ID_TAG = "id";

	private static final Log logger = LogFactory.getLog(TweetTagExtractor.class);

	private final ObjectMapper objectMapper;

	private String tagName;

	public TweetTagExtractor(String tagName) {
		Assert.notNull(tagName, "Tag name can not be null!");
		this.tagName = tagName;
		this.objectMapper = new ObjectMapper();
		Assert.notNull(objectMapper, "Failed to initialize the objectMapper");
	}

	@Override
	public String apply(Object input) {

		try {

			Map tweetJsonMap = null;

			if (input instanceof byte[]) {
				tweetJsonMap = objectMapper.readValue((byte[]) input, Map.class);
			}
			else if (input instanceof String) {
				tweetJsonMap = objectMapper.readValue((String) input, Map.class);
			}
			else if (input instanceof Map) {
				tweetJsonMap = (Map) input;
			}

			Assert.notNull(tweetJsonMap, "Failed to parse the Tweet json!");

			String tweetText = (String) tweetJsonMap.get(tagName);

			if (isEmpty(tweetText)) {
				logger.warn("Tweet with out [" + tagName + "] from tweet: " + tweetJsonMap.get(TWEET_ID_TAG));
				tweetText = "";
			}
			return tweetText;
		}
		catch (IOException e) {
			throw new RuntimeException("Can't parse input tweet json: " + input);
		}
	}
}
