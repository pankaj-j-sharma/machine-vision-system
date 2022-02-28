import { Routes } from "@angular/router";
import { AugmentationComponent } from "src/app/components/augmentation/augmentation.component";
import { DashboardComponent } from "src/app/components/dashboard/dashboard.component";
import { InsightsComponent } from "src/app/components/insights/insights.component";
import { LabellingComponent } from "src/app/components/labelling/labelling.component";
import { PredictionComponent } from "src/app/components/prediction/prediction.component";
import { ProcessTrackingComponent } from "src/app/components/process-tracking/process-tracking.component";
import { ProjectComponent } from "src/app/components/project/project.component";
import { TrainingComponent } from "src/app/components/training/training.component";
import { UserProfileComponent } from "src/app/components/user-profile/user-profile.component";

export const BULayoutRoutes: Routes = [
    { path: "bu-dashboard", component: DashboardComponent },
    { path: "bu-prediction", component: PredictionComponent },
    { path: "bu-insights", component: InsightsComponent },
    { path: 'bu-profile', component: UserProfileComponent },
    { path: "processtracking", component: ProcessTrackingComponent },

]  