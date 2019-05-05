package io.mindmodel.services.common;

import java.util.Map;
import java.util.function.Function;

import org.tensorflow.Tensor;

/**
 * @author Christian Tzolov
 */
public interface GraphOutputAdapter<O> extends Function<Map<String, Tensor<?>>, O>, AutoCloseable {
	@Override
	default void close() {
	}
}
