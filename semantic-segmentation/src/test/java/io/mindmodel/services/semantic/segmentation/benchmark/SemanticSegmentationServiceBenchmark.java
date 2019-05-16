/*
 * Copyright 2018 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance fromMemory the License.
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
package io.mindmodel.services.semantic.segmentation.benchmark;

import java.io.IOException;

import io.mindmodel.services.common.attic.GraphicsUtils;
import io.mindmodel.services.semantic.segmentation.SegmentationColorMap;
import io.mindmodel.services.semantic.segmentation.SemanticSegmentation;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.runner.RunnerException;

/**
 * @author Christian Tzolov
 */
public class SemanticSegmentationServiceBenchmark {

	@State(Scope.Benchmark)
	public static class ExecutionPlan {

		private SemanticSegmentation segmentationService;
		private byte[] inputImage;

		@Setup(Level.Trial)
		public void setUp() throws IOException {

			//String MODEL_URI = "http://download.tensorflow.org/models/xception_65_coco_pretrained_2018_10_02.tar.gz#frozen_inference_graph.pb";
			String MODEL_URI = "http://download.tensorflow.org/models/deeplabv3_mnv2_dm05_pascal_trainaug_2018_10_01.tar.gz#frozen_inference_graph.pb";
			segmentationService = new SemanticSegmentation(MODEL_URI, SegmentationColorMap.ADE20K_COLORMAP, null, 0.45f);

			inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/VikiMaxiAdi.jpg");
		}
	}

	@Fork(value = 1, warmups = 1)
	@Benchmark
	@BenchmarkMode(Mode.AverageTime)
	@Threads(value = 1)
	public void semanticSegmentationMaskImage(ExecutionPlan plan) {
		byte[] result = plan.segmentationService.maskImage(plan.inputImage);
	}

	public static void main(String[] args) throws IOException, RunnerException {
		org.openjdk.jmh.Main.main(args);
	}
}
