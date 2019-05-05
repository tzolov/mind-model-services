package io.mindmodel.services.twitter.sentiment.examples;

import io.mindmodel.services.common.attic.JsonMapperFunction;
import io.mindmodel.services.twitter.sentiment.TwitterSentimentService;
import io.mindmodel.services.twitter.sentiment.domain.SentimentResult;

/**
 * @author Christian Tzolov
 */
public class TwitterSentimentExample {

	public static void main(String[] args) {
		String tweet = "{\"text\": \"This is really bad\", \"id\":666, \"lang\":\"en\" }";

		TwitterSentimentService twitterSentimentService = new TwitterSentimentService(
				"http://dl.bintray.com/big-data/generic/minimal_graph.proto",
				"http://dl.bintray.com/big-data/generic/vocab.csv",
				true);

		SentimentResult tweetSentiment = twitterSentimentService.tweetSentiment(tweet);

		System.out.println(tweetSentiment.getSentiment() + " : " + tweetSentiment.getEstimate());

		String jsonTweetSentiment = new JsonMapperFunction().apply(tweetSentiment);
		System.out.println(jsonTweetSentiment);
	}
}
