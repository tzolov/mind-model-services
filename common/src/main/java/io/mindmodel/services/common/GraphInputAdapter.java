package io.mindmodel.services.common;

import java.util.Map;
import java.util.function.Function;

import org.tensorflow.Tensor;

/**
 * @author Christian Tzolov
 */
public interface GraphInputAdapter<I> extends Function<I, Map<String, Tensor<?>>>, AutoCloseable {
	@Override
	default void close() {
	}
}
