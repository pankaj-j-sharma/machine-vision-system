import { Routes } from "@angular/router";
import { ApiEndpointsComponent } from "src/app/components/api-endpoints/api-endpoints.component";
import { AugmentationComponent } from "src/app/components/augmentation/augmentation.component";
import { DashboardComponent } from "src/app/components/dashboard/dashboard.component";
import { DataSamplingComponent } from "src/app/components/data-sampling/data-sampling.component";
import { EvaluationComponent } from "src/app/components/evaluation/evaluation.component";
import { InsightsComponent } from "src/app/components/insights/insights.component";
import { LabellingComponent } from "src/app/components/labelling/labelling.component";
import { PredictionComponent } from "src/app/components/prediction/prediction.component";
import { ProcessTrackingComponent } from "src/app/components/process-tracking/process-tracking.component";
import { ProjectComponent } from "src/app/components/project/project.component";
import { TrainingComponent } from "src/app/components/training/training.component";
import { UserProfileComponent } from "src/app/components/user-profile/user-profile.component";

export const DSLayoutRoutes: Routes = [
    { path: "ds-dashboard", component: DashboardComponent },
    { path: "ds-prediction", component: PredictionComponent },
    { path: "ds-training", component: TrainingComponent },
    { path: "ds-labelling", component: LabellingComponent },
    { path: "ds-project", component: ProjectComponent },
    { path: "ds-augmentation", component: AugmentationComponent },
    { path: "ds-insights", component: InsightsComponent },
    { path: "ds-sampling", component: DataSamplingComponent },
    { path: "ds-process-tracking", component: ProcessTrackingComponent },
    { path: "ds-evaluation", component: EvaluationComponent },
    { path: 'ds-profile', component: UserProfileComponent },
    { path: 'ds-api-endpoints', component: ApiEndpointsComponent },

]  