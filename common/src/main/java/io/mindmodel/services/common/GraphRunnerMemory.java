package io.mindmodel.services.common;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

import io.mindmodel.services.common.util.AutoCloseables;
import org.tensorflow.Tensor;

/**
 * Keeps all tensorMap input parameters.
 */
public class GraphRunnerMemory implements Function<Map<String, Tensor<?>>, Map<String, Tensor<?>>>, AutoCloseable {

	private Map<String, Tensor<?>> tensorMap = new ConcurrentHashMap<>();

	public Map<String, Tensor<?>> getTensorMap() {
		return tensorMap;
	}

	@Override
	public Map<String, Tensor<?>> apply(Map<String, Tensor<?>> tensorMap) {
		this.tensorMap.putAll(tensorMap);
		return tensorMap;
	}

	@Override
	public void close() {
		AutoCloseables.all(this.tensorMap);
		this.tensorMap.clear();
	}
}
