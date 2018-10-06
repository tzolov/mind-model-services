/*
 * Copyright 2018 the original author or authors.
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
package io.mindmodel.services.semantic.segmentation;

import java.util.Arrays;
import java.util.function.Function;

import io.mindmodel.services.common.TensorFlowService;

import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.Resource;

/**
 * @author Christian Tzolov
 */
public class SemanticSegmentationService {

	private final SemanticSegmentationConfiguration segmentationConfiguration;
	private final Function<byte[], long[][]> segmentationFunction;

	public SemanticSegmentationConfiguration getSegmentationConfiguration() {
		return segmentationConfiguration;
	}

	public SemanticSegmentationService(String modelUri, boolean cachedModel) {
		this(new SemanticSegmentationConfiguration(),
				new TensorFlowService(new DefaultResourceLoader().getResource(modelUri),
						Arrays.asList(SemanticSegmentationUtils.OUTPUT_TENSOR_NAME), cachedModel));
	}

	public SemanticSegmentationService(SemanticSegmentationConfiguration segmentationConfiguration,
			TensorFlowService tensorFlowService) {
		this.segmentationConfiguration = segmentationConfiguration;
		this.segmentationFunction = segmentationConfiguration.inputConverter()
				.andThen(tensorFlowService).andThen(segmentationConfiguration.outputConverter());
	}

	public long[][] maskPixels(byte[] image) {
		return segmentationFunction.apply(image);
	}

	public byte[] augment(byte[] image) {
		return segmentationConfiguration.imageAugmenter().apply(image, segmentationFunction.apply(image));
	}

	public byte[] masksAsImage(byte[] image) {
		return segmentationConfiguration.pixelsToMaskImage().apply(segmentationFunction.apply(image));
	}
}
