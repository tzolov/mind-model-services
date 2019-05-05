package io.mindmodel.services.common;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.tensorflow.Session;
import org.tensorflow.Tensor;

import org.springframework.util.Assert;

/**
 * @author Christian Tzolov
 */
public class GraphRunner extends AutoCloseableSession
		implements Function<Map<String, Tensor<?>>, Map<String, Tensor<?>>> {

	/**
	 * Names expected in the named Tensor inside the input {@link GraphRunner#apply(Map)}.
	 * If the apply method will fail if the input map is missing some of the feedNames.
	 */
	private final List<String> feedNames;

	/**
	 * Names expected {@link GraphRunner#apply(Map)} result map.
	 */
	private final List<String> fetchNames;

	private final boolean autoCloseFeedTensors;

	public GraphRunner(String feedName, String fetchedName) {
		this(feedName, fetchedName, true);
	}

	public GraphRunner(List<String> feedNames, List<String> fetchedNames) {
		this(feedNames, fetchedNames, true);
	}

	public GraphRunner(String feedName, String fetchedName, boolean autoCloseFeedTensors) {
		this(Arrays.asList(feedName), Arrays.asList(fetchedName), autoCloseFeedTensors);
	}

	public GraphRunner(List<String> feedNames, List<String> fetchedNames, boolean autoCloseFeedTensors) {
		this.feedNames = feedNames;
		this.fetchNames = fetchedNames;
		this.autoCloseFeedTensors = autoCloseFeedTensors;
	}

	@Override
	public Map<String, Tensor<?>> apply(Map<String, Tensor<?>> feeds) {

		try {
			Assert.isTrue(feeds.keySet().containsAll(feedNames),
					"Applied feeds:" + feeds.keySet()
							+ "\n, don't match the expected feeds contract:" + this.feedNames);

			Session.Runner runner = this.getSession().runner();

			// Feed in the input named tensors
			for (Map.Entry<String, Tensor<?>> e : feeds.entrySet()) {
				String feedName = e.getKey();
				runner = runner.feed(feedName, e.getValue());
			}

			// Set the tensor name to be fetched after the evaluation
			for (String fetchName : this.fetchNames) {
				runner.fetch(fetchName);
			}

			// Evaluate the input
			List<Tensor<?>> outputTensors = runner.run();

			// Extract the output tensors
			Map<String, Tensor<?>> outTensorMap = new HashMap<>();
			for (int outputIndex = 0; outputIndex < this.fetchNames.size(); outputIndex++) {
				outTensorMap.put(this.fetchNames.get(outputIndex), outputTensors.get(outputIndex));
			}

			return outTensorMap;
		}
		finally {
			if (this.autoCloseFeedTensors) {
				for (Tensor<?> t : feeds.values()) {
					t.close();
				}
			}
		}
	}

	public List<String> getFeedNames() {
		return feedNames;
	}

	public String getFeedName() {
		Assert.isTrue(feedNames.size() == 1, "Assumes a single feed input");
		return feedNames.get(0);
	}

	public List<String> getFetchNames() {
		return fetchNames;
	}

	public String getFetchName() {
		Assert.isTrue(fetchNames.size() == 1, "Assumes a single fetch output");
		return fetchNames.get(0);
	}

	public boolean isAutoCloseFeedTensors() {
		return autoCloseFeedTensors;
	}
}
