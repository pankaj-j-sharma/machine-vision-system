import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BuLayoutComponent } from './bu-layout.component';

describe('BuLayoutComponent', () => {
  let component: BuLayoutComponent;
  let fixture: ComponentFixture<BuLayoutComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ BuLayoutComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(BuLayoutComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
