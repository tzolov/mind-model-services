package io.mindmodel.services.common;

import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.op.Ops;

import org.springframework.core.io.Resource;

/**
 * @author Christian Tzolov
 */
public class ImportedGraphRunner extends GraphRunner {

	/**
	 * Location of the pre-trained model archive.
	 */
	private Resource modelLocation;

	/**
	 * If set true the pre-trained model is cached on the local file system.
	 */
	private boolean cacheModel;

	public ImportedGraphRunner(Resource modelLocation, List<String> feedNames,
			List<String> fetchNames, boolean cacheModel) {
		this(modelLocation, feedNames, fetchNames, cacheModel, true);
	}

	/**
	 *
	 * @param modelLocation Location of the pre-trained model archive.
	 * @param feedNames
	 * @param fetchNames
	 * @param cacheModel If set true the pre-trained model is cached on the local file system.
	 * @param autoCloseFeedTensors If true the feed tensors are closed after each evaluations.
	 */
	public ImportedGraphRunner(Resource modelLocation, List<String> feedNames,
			List<String> fetchNames, boolean cacheModel, boolean autoCloseFeedTensors) {
		super(feedNames, fetchNames, autoCloseFeedTensors);
		this.modelLocation = modelLocation;
		this.cacheModel = cacheModel;
	}

	@Override
	protected void doGraphDefinition(Ops tf) {
		// Extract the pre-trained model as byte array.
		byte[] model = this.cacheModel ? new CachedModelExtractor().getModel(this.modelLocation)
				: new ModelExtractor().getModel(this.modelLocation);
		// Import the pre-trained model
		((Graph) tf.scope().env()).importGraphDef(model);
	}
}
