import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { HomeComponent } from './components/home/home.component';
import { LoginComponent } from './components/login/login.component';
import { BuLayoutComponent } from './layouts/bu-layout/bu-layout.component';
import { DsLayoutComponent } from './layouts/ds-layout/ds-layout.component';

const routes: Routes = [
  {
    path: "",
    redirectTo: "login",
    pathMatch: "full"
  },

  { path: 'login', component: LoginComponent },
  { path: 'home', component: HomeComponent },
  {
    path: "",
    component: DsLayoutComponent,
    children: [
      {
        path:"",
        loadChildren:()=> import('./layouts/ds-layout/ds-layout.module').then(m=>m.DSLayoutModule)
      }
    ]
  },  
  
  {
    path: "",
    component: BuLayoutComponent,
    children: [
      {
        path:"",
        loadChildren:()=> import('./layouts/bu-layout/bu-layout.module').then(m=>m.BULayoutModule)
      }
    ]
  },  

  {
    path: "**",
    redirectTo: "login",
  },

];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
