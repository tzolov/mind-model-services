/*
 * Copyright 2017-2018 the original author or authors.
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

package io.mindmodel.services.common;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Function;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;

import org.springframework.core.io.Resource;

/**
 * @author Christian Tzolov
 */
public class TensorFlowService implements AutoCloseable, Function<Map<String, Tensor>, Map<String, Tensor<?>>> {

	private static final Log logger = LogFactory.getLog(TensorFlowService.class);

	private Graph graph;

	private List<String> fetchedNames;

	public TensorFlowService(Resource modelLocation, List<String> fetchedNames) {
		this(modelLocation, fetchedNames, false);
	}

	public TensorFlowService(Resource modelLocation, List<String> fetchedNames, boolean cacheModel) {
		if (logger.isInfoEnabled()) {
			logger.info("Loading TensorFlow graph model: " + modelLocation);
		}
		this.fetchedNames = fetchedNames;
		this.graph = new Graph();
		byte[] model = cacheModel? new CachedModelExtractor().getModel(modelLocation) : new ModelExtractor().getModel(modelLocation);
		this.graph.importGraphDef(model);
	}

	/**
	 * Evaluates a pre-trained tensorflow model (encoded as {@link Graph}). Use the feeds parameter to feed in the
	 * model input data and fetch-names to specify the output tensors.
	 *
	 * @param feeds Named map of input tensors.
	 * @return Returns the computed output tensors. The names of the output tensors is defined by the fetchedNames
	 * argument
	 */
	@Override
	public Map<String, Tensor<?>> apply(Map<String, Tensor> feeds) {

		try (Session session = new Session(graph)) {

			Runner runner = session.runner();

			// Keep tensor references to release them in the finally block
			Tensor[] feedTensors = new Tensor[feeds.size()];
			try {
				// Feed in the input named tensors
				int inputIndex = 0;
				for (Entry<String, Tensor> e : feeds.entrySet()) {
					String feedName = e.getKey();
					feedTensors[inputIndex] = e.getValue();
					runner = runner.feed(feedName, feedTensors[inputIndex]);
					inputIndex++;
				}

				// Set the tensor name to be fetched after the evaluation
				for (String fetchName : this.fetchedNames) {
					runner.fetch(fetchName);
				}

				// Evaluate the input
				List<Tensor<?>> outputTensors = runner.run();

				// Extract the output tensors
				Map<String, Tensor<?>> outTensorMap = new HashMap<>();
				for (int outputIndex = 0; outputIndex < this.fetchedNames.size(); outputIndex++) {
					outTensorMap.put(this.fetchedNames.get(outputIndex), outputTensors.get(outputIndex));
				}
				return outTensorMap;
			}
			finally {
				// Release all feed tensors
				for (Tensor tensor : feedTensors) {
					if (tensor != null) {
						tensor.close();
					}
				}
			}
		}
	}

	@Override
	public void close() {
		logger.info("Close TensorFlow Graph!");
		if (graph != null) {
			graph.close();
		}
	}
}
