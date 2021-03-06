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

package io.mindmodel.services.twitter.sentiment.domain;

/**
 * @author Christian Tzolov
 */
public enum Sentiment {

	POSITIVE, NEUTRAL, NEGATIVE;

	public static Sentiment get(float estimate) {
		if (estimate > 0.5) {
			return POSITIVE;
		}
		else if (estimate < 0.4) {
			return NEGATIVE;
		}

		return NEUTRAL;
	}

}
