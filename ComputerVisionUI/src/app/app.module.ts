import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import {NgbModule} from '@ng-bootstrap/ng-bootstrap';
import { LoginComponent } from './components/login/login.component';
import { HomeComponent } from './components/home/home.component';
import { FormsModule, ReactiveFormsModule } from "@angular/forms";
import { HttpClientModule, HTTP_INTERCEPTORS } from '@angular/common/http';
import { RegisterComponent } from './components/register/register.component';
import { NgbAlertModule } from '@ng-bootstrap/ng-bootstrap';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';

import { ApiInterceptorService } from './services/apiInterceptor/api-interceptor.service';

import { DsLayoutComponent } from './layouts/ds-layout/ds-layout.component';

import { DashboardComponent } from './components/dashboard/dashboard.component';
import { FooterComponent } from './components/footer/footer.component';
import { NavbarComponent } from './components/navbar/navbar.component';
import { BuLayoutComponent } from './layouts/bu-layout/bu-layout.component';
import { PredictionComponent } from './components/prediction/prediction.component';
import { TrainingComponent } from './components/training/training.component';
import { LabellingComponent } from './components/labelling/labelling.component';
import { ProjectComponent } from './components/project/project.component';
import { AugmentationComponent } from './components/augmentation/augmentation.component';
import { InsightsComponent } from './components/insights/insights.component';
import { DataSamplingComponent } from './components/data-sampling/data-sampling.component';
import { FileUploadComponent } from './components/_shared/file-upload/file-upload.component';
import { ModalPopupComponent } from './components/_shared/modal-popup/modal-popup.component';
import { ProcessTrackingComponent } from './components/process-tracking/process-tracking.component';
import { EvaluationComponent } from './components/evaluation/evaluation.component';
import { AppChartComponent } from './components/_shared/app-chart/app-chart.component';
import { UserProfileComponent } from './components/user-profile/user-profile.component';
import { CanvasComponent } from './components/labelling/canvas-component';
import { NgxEchartsModule } from 'ngx-echarts';
import { DataConnectorsComponent } from './components/_shared/data-connectors/data-connectors.component';
import { ApiEndpointsComponent } from './components/api-endpoints/api-endpoints.component';


@NgModule({
  declarations: [
    AppComponent,
    LoginComponent,
    HomeComponent,
    RegisterComponent,
    DsLayoutComponent,
    FooterComponent, 
    NavbarComponent,    
    DashboardComponent, BuLayoutComponent, PredictionComponent, TrainingComponent, LabellingComponent, ProjectComponent, AugmentationComponent, InsightsComponent, DataSamplingComponent, FileUploadComponent, ModalPopupComponent, ProcessTrackingComponent, EvaluationComponent, AppChartComponent, UserProfileComponent,
    CanvasComponent,
    DataConnectorsComponent,
    ApiEndpointsComponent
  ],
  imports: [
    BrowserModule,
    CommonModule,
    RouterModule,
    AppRoutingModule,
    NgbModule,
    FormsModule,
    ReactiveFormsModule,
    HttpClientModule,
    NgbAlertModule,
    NgxEchartsModule.forRoot({
      echarts: () => import('echarts')
    })    
  ],
  exports: [FooterComponent, NavbarComponent,DsLayoutComponent],
  providers: [
    { provide: HTTP_INTERCEPTORS, useClass: ApiInterceptorService, multi: true }
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
