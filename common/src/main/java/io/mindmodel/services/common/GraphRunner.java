package io.mindmodel.services.common;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;

import org.springframework.util.Assert;

/**
 * @author Christian Tzolov
 */
public class GraphRunner extends AutoCloseableSession
		implements Function<Map<String, Tensor<?>>, Map<String, Tensor<?>>> {

	/**
	 * Interface providing Graph's model.
	 */
	private GraphDefinition graphDefinition;

	/**
	 * Names expected in the named Tensor inside the input {@link GraphRunner#apply(Map)}.
	 * If the apply method will fail if the input map is missing some of the feedNames.
	 */
	private final List<String> feedNames;

	/**
	 * Names expected {@link GraphRunner#apply(Map)} result map.
	 */
	private final List<String> fetchNames;

	/**
	 * When set and the input takes a single feed, then the name of the input tensor is automatically mapped
	 * to the expected input name. E.g. no need to rename the input names explicitly.
	 */
	private boolean autoBinding;

	public GraphRunner(String feedName, String fetchedName) {
		this(Arrays.asList(feedName), Arrays.asList(fetchedName));
	}

	public GraphRunner(List<String> feedNames, List<String> fetchedNames) {
		this.feedNames = feedNames;
		this.fetchNames = fetchedNames;
		this.autoBinding = feedNames.size() == 1;
	}

	@Override
	public Map<String, Tensor<?>> apply(Map<String, Tensor<?>> feeds) {

		Assert.notNull(this.graphDefinition, "Graph Definition is compulsory!");

		if (!this.isAutoBinding() && !feeds.keySet().containsAll(this.feedNames)) {
			throw new IllegalArgumentException("Applied feeds:" + feeds.keySet()
					+ "\n, don't match the expected feeds contract:" + this.feedNames);
		}

		if (this.isAutoBinding() && (feeds.size() != 1)) {
			throw new IllegalArgumentException("Feed auto-binding expects a " +
					"single feed tensors but found: " + feeds);
		}

		Session.Runner runner = this.getSession().runner();

		// Feed in the input named tensors
		for (Map.Entry<String, Tensor<?>> feedEntry : feeds.entrySet()) {
			String feedName = (this.isAutoBinding()) ? this.feedNames.get(0) : feedEntry.getKey();
			runner = runner.feed(feedName, feedEntry.getValue());
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

	public List<String> getFeedNames() {
		return this.feedNames;
	}

	public String getSingleøøƶFeedName() {
		Assert.isTrue(feedNames.size() == 1, "Assumes a single feed input");
		return this.feedNames.get(0);
	}

	public List<String> getFetchNames() {
		return this.fetchNames;
	}

	public String getSingleFetchName() {
		Assert.isTrue(this.fetchNames.size() == 1, "Assumes a single fetch output");
		return this.fetchNames.get(0);
	}

	public boolean isAutoBinding() {
		return this.autoBinding;
	}

	public GraphRunner disableAutoBinding() {
		this.autoBinding = false;
		return this;
	}

	public GraphRunner enableAutoBinding() {
		if (this.getFeedNames().size() != 1) {
			throw new IllegalArgumentException("Auto-binding is permitted for Graphs with single input feed, but " +
					" found: " + this.getFeedNames());
		}
		this.autoBinding = true;
		return this;
	}

	public GraphRunner withGraphDefinition(GraphDefinition graphDefinition) {
		this.graphDefinition = graphDefinition;
		return this;
	}

	@Override
	protected void doGraphDefinition(Ops tf) {
		this.graphDefinition.defineGraph(tf);
	}

	@Override
	public String toString() {
		return String.format("(%s) -> (%s)", String.join(",", this.feedNames),
				String.join(",", this.fetchNames));
	}
}
