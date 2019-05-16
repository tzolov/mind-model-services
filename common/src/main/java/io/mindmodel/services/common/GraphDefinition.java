package io.mindmodel.services.common;

import org.tensorflow.op.Ops;

/**
 * @author Christian Tzolov
 */
public interface GraphDefinition {
	void defineGraph(Ops tf);
}
