package io.mindmodel.services.pose.estimation;

import java.util.Collections;
import java.util.List;
import java.util.function.Function;

import io.mindmodel.services.common.TensorFlowService;
import io.mindmodel.services.pose.estimation.domain.Body;

import org.springframework.core.io.Resource;
import org.springframework.util.CollectionUtils;

/**
 * @author Christian Tzolov
 */
public class PoseEstimationService {

	private final Function<byte[][], List<List<Body>>> poseEstimationFunction;

	private final PoseEstimationTensorflowInputConverter inputConverter;
	private final PoseEstimationTensorflowOutputConverter outputConverter;
	private final TensorFlowService tfService;

	public PoseEstimationService(Resource modelResource, List<String> fetchNames) {
		this(new PoseEstimationTensorflowInputConverter(), new PoseEstimationTensorflowOutputConverter(fetchNames),
				new TensorFlowService(modelResource, fetchNames));
	}

	public PoseEstimationService(PoseEstimationTensorflowInputConverter inputConverter,
			PoseEstimationTensorflowOutputConverter outputConverter, TensorFlowService tfService) {
		this.inputConverter = inputConverter;
		this.outputConverter = outputConverter;
		this.tfService = tfService;

		this.poseEstimationFunction = this.inputConverter.andThen(this.tfService).andThen(this.outputConverter);
	}

	public List<Body> detect(byte[] image) {
		List<List<Body>> batches = this.poseEstimationFunction.apply(new byte[][] { image });
		if (CollectionUtils.isEmpty(batches)) {
			return Collections.emptyList();
		}
		return batches.get(0);
	}

	public List<List<Body>> detect(byte[][] images) {
		return this.poseEstimationFunction.apply(images);
	}

	public PoseEstimationTensorflowInputConverter getInputConverter() {
		return inputConverter;
	}

	public PoseEstimationTensorflowOutputConverter getOutputConverter() {
		return outputConverter;
	}

	public TensorFlowService getTfService() {
		return tfService;
	}
}
