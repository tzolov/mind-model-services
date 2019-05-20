package io.mindmodel.services.common;

import io.mindmodel.services.common.util.CachedModelExtractor;
import io.mindmodel.services.common.util.ModelExtractor;
import org.tensorflow.Graph;
import org.tensorflow.op.Ops;

import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.Resource;

/**
 * @author Christian Tzolov
 */
public class ProtoBufGraphDefinition implements GraphDefinition {

	/**
	 * Location of the pre-trained model archive.
	 */
	private final Resource modelLocation;

	/**
	 * If set true the pre-trained model is cached on the local file system.
	 */
	private final boolean cacheModel;

	public ProtoBufGraphDefinition(String modelUri, boolean cacheModel) {
		this(new DefaultResourceLoader().getResource(modelUri), cacheModel);
	}

	public ProtoBufGraphDefinition(Resource modelLocation, boolean cacheModel) {
		this.modelLocation = modelLocation;
		this.cacheModel = cacheModel;
	}

	@Override
	public void defineGraph(Ops tf) {
		// Extract the pre-trained model as byte array.
		byte[] model = this.cacheModel ? new CachedModelExtractor().getModel(this.modelLocation)
				: new ModelExtractor().getModel(this.modelLocation);
		// Import the pre-trained model
		((Graph) tf.scope().env()).importGraphDef(model);
	}
}
